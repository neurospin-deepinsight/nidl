import inspect
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch

MetricsType = Callable | list[Callable] | dict[str, Callable]

NeedsType = (
    list[str | Callable]
    | dict[str, str | Callable]
    | dict[str, list[str | Callable] | dict[str, str | Callable]]
    | None
)


class MetricsCallback(pl.Callback):
    """Callback to compute and log metrics during training, validation and test
    of a PL model.

    This callback will:

    1) Collect the model's outputs after each training/validation/test step.
    2) Compute the required metrics on the collected outputs, either batch-wise
       or epoch-wise (depending on the use-case).
    3) Log the metrics after each iteration or epoch.

    It handles multi-GPU distributed training setups. It is compatible with
    :class:`torchmetrics` metrics, scikit-learn metrics and custom metric
    functions for metrics computation.

    *Important:* we assume that the model's step methods (training_step,
    validation_step, test_step) return a dict of outputs that contain all the
    necessary information to compute the metrics. Some keys in this dict should
    match those specified in the `needs` argument. The values should be
    tensors or numpy arrays.

    Parameters
    ----------
    metrics: dict[str, Callable] or list[Callable] or Callable
        A list of metrics (callables including :class:`torchmetrics.Metric`
        and sklearn metrics) to be computed on the model's outputs.
        If a dict is provided, the keys will be used as the metric names
        when logging. If a Callable or list of Callable is provided, the metric
        names will be inferred from the function or class names.

    needs: NeedsType, default=None
        A mapping defining which outputs from the model are needed to compute
        the metrics. It can be either:

        - None: the needed arguments are inferred automatically from the
          metric signatures and parsed from the model outputs. It will throw an
          error if the required arguments are not found in the outputs or if
          there are ambiguities.

        - List[str | Callable] or dict[str, str | Callable]:
          applies to ALL metrics, e.g.
          * ["logits", "labels"] (positional arguments) if the metric
            needs "preds" and "targets" as positional arguments, and these are
            found in the model outputs under the keys "logits" and "labels".

          * {
                "preds": lambda outputs: outputs["logits"].softmax(dim=-1),
                "targets": "labels"
            }
            if the metric needs "preds" and "targets" as keyword arguments, and
            these are found in the model outputs under the keys "logits" (with
            pre-processing required) and "labels" (used as-is) respectively.

        - dict[str, List[str | Callable] | dict[str, str | Callable]]:
          per-metric overrides, keyed by metric name, e.g.
          {
            "Accuracy": [
                lambda outputs: outputs["logits"].softmax(dim=-1),
                "labels"
                ],
            "Uniformity": {"z1": "Z1", "z2": "Z2"}
          }
          if different metrics need different arguments from the outputs. The
          same logic applies per-metric as above.

    compute_per_step: bool, default=False
        Ignored for :class:`torchmetrics.Metric` instances, which always handle
        per-step updates internally. For other metrics (e.g. sklearn metrics or
        custom functions), whether to compute the metrics at each step (batch)
        or only at the end of the epoch. If True, metrics are computed at each
        step and averaged at the end of the epoch. This is useful for metrics
        that can be computed batch-wise (e.g. accuracy). If False, all needed
        outputs are collected and the metric is computed only once, ensuring
        exact results but requiring more memory.

    compute_on_cpu: bool, default=True
        Whether to move the collected outputs to CPU for metrics computation.
        This is useful to avoid GPU memory issues when using metrics that
        require all outputs to be in memory at once (i.e.
        `compute_per_step=False`).

    every_n_train_steps: Union[int, None], default=1
        Frequency (in training steps) to compute and log metrics during
        training. If 0 or None, metrics are not computed during training. This
        is mutually exclusive with `every_n_train_epochs`.

    every_n_train_epochs: Union[int, None], default=None
        Frequency (in epochs) to compute and log metrics during training.
        If 0 or None, metrics are not computed during training. This is
        mutually exclusive with `every_n_train_steps`.

    every_n_val_epochs: Union[int, None], default=1
        Frequency (in epochs) to compute and log metrics during validation.
        If 0 or None, metrics are not computed during validation.

    on_test_end: bool, default=False
        Whether to compute and log metrics at the end of the test.

    prog_bar: bool, default=True
        Whether to display the metrics in the progress bar.

    Examples
    --------
    A simple use-case for classification metrics during training and
    validation. We assume that the model's training_step and validation_step
    return the logits as "preds" and targets as "targets" in their outputs
    dictionary:

    >>> from nidl.callbacks import MetricsCallback
    >>> from torchmetrics.metrics import Accuracy, F1Score
    >>> metrics = {"acc": Accuracy(), "f1": F1Score()}
    >>> metrics_callback = MetricsCallback(
    ...     metrics=metrics,
    ...     needs=["preds", "targets"], # applies to all metrics
    ...     every_n_train_epochs=1,
    ...     every_n_val_epochs=1,
    ...     on_test_end=True,
    ... )

    Another use-case for self-supervised metrics during training only. We
    assume that the model's training_step returns the embeddings "Z1" and "Z2"
    in its outputs dictionary and the metrics require keyword arguments "z1"
    and "z2":

    >>> from nidl.callbacks import MetricsCallback
    >>> from nidl.metrics.ssl import Alignment, ContrastiveAccuracy, Uniformity
    >>> metrics = [ContrastiveAccuracy(), Alignment(), Uniformity()]
    >>> metrics_callback = MetricsCallback(
    ...     metrics=metrics,
    ...     needs={"z1": "Z1", "z2": "Z2"}, # applies to all metrics
    ...     every_n_train_epochs=1,
    ...     every_n_val_epochs=None,
    ...     on_test_end=False,
    ... )
    """

    def __init__(
        self,
        metrics: MetricsType,
        needs: NeedsType = None,
        compute_per_step: bool = False,
        compute_on_cpu=True,
        every_n_train_steps: int | None = 1,
        every_n_train_epochs: int | None = None,
        every_n_val_epochs: int | None = 1,
        on_test_end: bool = False,
        prog_bar=True,
    ):
        super().__init__()

        self.metrics = metrics
        self.needs = needs
        self.every_n_train_steps = every_n_train_steps
        self.every_n_train_epochs = every_n_train_epochs
        self.every_n_val_epochs = every_n_val_epochs
        self._on_test_end = on_test_end
        self.prog_bar = prog_bar

        self.counter_val_epochs = 0

        self._train_collector = MetricsCollection(
            self.metrics,
            self.needs,
            compute_per_step=compute_per_step,
            compute_on_cpu=compute_on_cpu,
        )
        self._val_collector = MetricsCollection(
            self.metrics,
            self.needs,
            compute_per_step=compute_per_step,
            compute_on_cpu=compute_on_cpu,
        )
        self._test_collector = MetricsCollection(
            self.metrics,
            self.needs,
            compute_per_step=compute_per_step,
            compute_on_cpu=compute_on_cpu,
        )

    def compute_metrics_and_log(
        self,
        trainer,
        pl_module,
        collector,
        on_step,
        on_epoch,
        reset=True,
    ):
        """Compute and log metrics using the collected outputs."""

        # Gather all tensors and compute metrics
        scalars = collector.compute(trainer)

        # Log the metrics on rank-zero only
        if trainer.is_global_zero:
            pl_module.log_dict(
                scalars,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=self.prog_bar,
                sync_dist=False,
            )

        if reset:
            collector.reset()  # free the cache

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        collect_train_outputs = (
            self.every_n_train_steps is not None
            and self.every_n_train_steps > 0
        ) or (  # only collect if we are computing at epoch end
            self.every_n_train_epochs is not None
            and self.every_n_train_epochs > 0
            and trainer.current_epoch % self.every_n_train_epochs == 0
        )
        if collect_train_outputs:
            self._train_collector.collect(outputs)

        if (
            self.every_n_train_steps is not None
            and self.every_n_train_steps > 0
        ):
            global_step = trainer.global_step
            if global_step % self.every_n_train_steps == 0:
                self.compute_metrics_and_log(
                    trainer,
                    pl_module,
                    self._train_collector,
                    on_step=True,
                    on_epoch=False,
                )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if getattr(trainer, "sanity_checking", False):
            return  # skip during sanity check

        collect_val_outputs = (  # only collect if we are logging at epoch end
            self.every_n_val_epochs is not None
            and self.every_n_val_epochs > 0
            and self.counter_val_epochs % self.every_n_val_epochs == 0
        )
        if collect_val_outputs:
            self._val_collector.collect(outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self._on_test_end:
            self._test_collector.collect(outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_train_epochs is not None
            and self.every_n_train_epochs > 0
            and trainer.current_epoch % self.every_n_train_epochs == 0
        ):
            self.compute_metrics_and_log(
                trainer,
                pl_module,
                self._train_collector,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.counter_val_epochs += 1
        if (
            self.every_n_val_epochs is not None
            and self.every_n_val_epochs > 0
            and self.counter_val_epochs % self.every_n_val_epochs == 0
        ):
            self.compute_metrics_and_log(
                trainer,
                pl_module,
                self._val_collector,
                on_step=False,
                on_epoch=True,
            )

    def on_test_epoch_end(self, trainer, pl_module):
        if self._on_test_end:
            self.compute_metrics_and_log(
                trainer,
                pl_module,
                self._test_collector,
                on_step=False,
                on_epoch=True,
            )


class MetricsCollection:
    """Helper class to collect outputs during training/validation/test and
    compute metrics.

    It handles the collection of outputs at each step and computation of
    metrics. It can handle either `torchmetrics.Metric` instances,
    scikit-learn metrics or custom metric functions.

    Notes
    -----
    If scikit-learn metrics or custom functions are used and `compute_per_step`
    is False, the entire outputs needed for metric computations must fit in
    memory (either CPU or GPU). We advise using `compute_on_cpu=True` in this
    case to avoid GPU memory issues.

    We recommend using `torchmetrics.Metric` instances whenever possible, as
    they handle memory efficiently in a distributed fashion.

    """

    def __init__(
        self,
        metrics: MetricsType,
        needs: NeedsType,
        compute_per_step: bool = False,
        compute_on_cpu=True,
    ):
        self.global_metrics_args = self.is_global_metrics_args(needs)
        self.metrics = self._parse_metrics(metrics)
        self.needs = self._parse_needs(needs, self.metrics)
        self.compute_per_step = compute_per_step
        self.compute_on_cpu = compute_on_cpu
        self.outputs_cache = {name: [] for name in self.metrics}
        if self.global_metrics_args:  # store outputs only once
            self.outputs_cache = []
        self.metrics_cache = {name: [] for name in self.metrics}

    def collect(self, outputs):
        """Collect outputs from a training/validation/test step."""
        if self.global_metrics_args:
            args, kwargs = self._parse_and_filter_outputs(outputs)
        global_outputs_cached = False
        for name, metric in self.metrics.items():
            if not self.global_metrics_args:
                (args, kwargs) = self._parse_and_filter_outputs(outputs, name)
            if self._is_torchmetric(metric):
                metric.update(*args, **kwargs)
            elif self.compute_per_step:
                self.metrics_cache[name].append(
                    self._parse_output(metric(*args, **kwargs))
                )
            else:
                # Collect the outputs for later computation
                # This is memory-intensive for large datasets!
                if self.global_metrics_args and not global_outputs_cached:
                    self.outputs_cache.append((args, kwargs))
                    global_outputs_cached = True
                else:
                    self.outputs_cache[name].append((args, kwargs))

    def compute(self, trainer):
        """Compute the metrics on the collected outputs.

        Parameters
        ----------
        trainer: pl.Trainer
            The PyTorch Lightning trainer instance.

        Returns
        -------
        scalars: dict {str: torch.Tensor}
            The metric values as a mapping of metric names to scalar tensors.
        """

        scalars = {}

        for name, metric in self.metrics.items():
            if self._is_torchmetric(metric):
                value = metric.compute()
            else:
                if self.compute_per_step:
                    # Average over step-wise computations
                    values = torch.stack(self.metrics_cache[name])
                    value = trainer.strategy.reduce(
                        values.mean(), reduce_op="mean"
                    )
                else:
                    (args, kwargs) = self.concat_outputs(
                        self.outputs_cache[name]
                    )
                    # Gather across devices before computing the metric
                    args = self.gather(trainer, args)
                    kwargs = self.gather(trainer, kwargs)
                    value = metric(*args, **kwargs)
            scalars[name] = value
        return scalars

    def reset(self):
        """Reset all caches."""
        self.outputs_cache = {name: [] for name in self.metrics}
        if self.global_metrics_args:  # store outputs only once
            self.outputs_cache = []
        self.metrics_cache = {name: [] for name in self.metrics}

        for metric in self.metrics.values():
            if self._is_torchmetric(metric):
                metric.reset()

    def concat_outputs(self, outputs_list):
        """Concatenate the collected outputs along the batch dimension."""
        if len(outputs_list) == 0:
            raise ValueError("No outputs collected to concatenate")

        first_args, first_kwargs = outputs_list[0]

        # Concatenate positional arguments
        args = []
        n_args = len(first_args)
        for idx in range(n_args):
            arg_tensors = [outputs[0][idx] for outputs in outputs_list]
            args.append(torch.cat(arg_tensors, dim=0))

        # Concatenate keyword arguments
        kwargs = {}
        for key in first_kwargs:
            kwarg_tensors = [outputs[1][key] for outputs in outputs_list]
            kwargs[key] = torch.cat(kwarg_tensors, dim=0)

        return args, kwargs

    def is_global_metrics_args(self, needs: NeedsType) -> bool:
        """Check if the `needs` mapping applies to all metrics globally."""
        if needs is None:
            return False
        return not (
            isinstance(needs, dict)
            and any(isinstance(v, dict) for v in needs.values())
        )

    def _parse_and_filter_outputs(self, outputs, metric_name=None):
        """Parse and filter the outputs according to the `needs` mapping."""
        if metric_name is None:
            # Global mapping for all metrics, take the first one
            metric_name = next(iter(self.metrics))
        mapping = self.needs[metric_name]
        if isinstance(mapping, list):
            # Positional arguments
            args = []
            for key_or_fn in mapping:
                if isinstance(key_or_fn, str):
                    if key_or_fn not in outputs:
                        raise KeyError(
                            f"Output key `{key_or_fn}` not found in model's "
                            f"outputs for metric `{metric_name}`"
                        )
                    value = outputs[key_or_fn]
                elif callable(key_or_fn):
                    value = key_or_fn(outputs)
                else:
                    raise TypeError(
                        f"Invalid type in `needs` list for metric "
                        f"`{metric_name}`: {type(key_or_fn)}"
                    )
                args.append(self._parse_output(value))
            return args, {}
        elif isinstance(mapping, dict):
            # Keyword arguments
            kwargs = {}
            for arg_name, key_or_fn in mapping.items():
                if isinstance(key_or_fn, str):
                    value = outputs[key_or_fn]
                elif callable(key_or_fn):
                    value = key_or_fn(outputs)
                else:
                    raise TypeError(
                        f"Invalid type in `needs` dict for metric "
                        f"`{metric_name}`: {type(key_or_fn)}"
                    )
                kwargs[arg_name] = self._parse_output(value)
            return [], kwargs
        else:
            raise TypeError(
                f"Invalid needs mapping for metric `{metric_name}`: "
                f"{type(mapping)}"
            )

    def _parse_output(self, output):
        """Parse and move output to the desired device."""
        if isinstance(output, torch.Tensor):
            t = output.detach()
        elif isinstance(output, np.ndarray):
            t = torch.from_numpy(output)
        else:
            raise ValueError(
                "Output value must be a torch.Tensor or np.ndarray,"
                f"got {type(output)}"
            )
        # If we are going to store (compute_per_step=False), move to CPU
        if self.compute_on_cpu and not self.compute_per_step:
            # BUT: only do this for caching; gather() will move back for DDP
            t = t.cpu()
        return t

    def _infer_args(self, fn: Callable) -> list[str]:
        """
        Return ordered argument names (positional-or-keyword + keyword-only).
        Raise if *args are present (cannot parse unambiguously) and warning if
        **kwargs are present.
        """
        target = fn.update if self._is_torchmetric(fn) else fn
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Cannot inspect signature of {target!r}; "
                "please provide `needs` explicitly."
            ) from None

        params = list(sig.parameters.values())
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
            raise RuntimeError(
                f"{target.__name__} has *args; cannot infer mapping.\n"
                "Provide `needs`."
            )
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            raise Warning(
                f"{target.__name__} has **kwargs; these arguments cannot be "
                "inferred.\n"
            )

        names = [
            p.name
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        return names

    def _validate_kw_mapping(
        self, name: str, mapping: dict[str, str | Callable]
    ) -> None:
        if not isinstance(mapping, dict):
            raise TypeError(
                f"`needs[{name}]` must be a dict of arg -> (str|Callable),"
                f"got {type(mapping)}"
            )
        for k, v in mapping.items():
            if not (isinstance(v, str) or callable(v)):
                raise TypeError(
                    f"`needs[{name}][{k}]` must be str or Callable,"
                    f"got {type(v)}"
                )

    def _parse_metrics(self, metrics: MetricsType) -> dict[str, Callable]:
        """Normalize metrics to {name: callable} with unique, stable names."""
        if callable(metrics):
            metrics = [metrics]
        if isinstance(metrics, list):
            if not metrics:
                raise ValueError("`metrics` list cannot be empty")
            out = {}
            for m in metrics:
                if not callable(m):
                    raise TypeError(
                        "All items in metrics list must be Callable"
                    )
                base = getattr(m, "__name__", m.__class__.__name__)
                name = base
                i = 1
                while name in out:
                    name = f"{base}_{i}"
                    i += 1
                out[name] = m
            return out
        if isinstance(metrics, dict):
            if not metrics:
                raise ValueError("`metrics` dict cannot be empty")
            for k, m in metrics.items():
                if not callable(m):
                    raise TypeError(f"metrics['{k}'] must be Callable")
            return dict(metrics)
        raise TypeError(
            "metrics must be Callable or list[Callable] or dict[str, Callable]"
        )

    def _parse_needs(
        self,
        needs: NeedsType,
        metrics: MetricsType,  # {metric_name: callable}
    ):
        """
        Normalize `needs` into:
        { metric_name:
            [ output_key | prepare_fn ]               # positional
            | { arg_name: (output_key | prepare_fn) }   # keyword
        }

        Rules:

        - If needs is None -> infer arg names and return keyword
          mapping {arg: arg}.
        - If needs is a GLOBAL positional list -> copy the same list to all
          metrics.
        - If needs is a GLOBAL keyword dict -> copy the same dict to all
          metrics.
        - If needs is a PER-METRIC dict -> take values as-is (list or dict) per
          metric.
        - No alignment/validation against metric signatures
          (user responsibility).
        """
        out = {}

        # Case A: needs is None -> infer KW mapping {arg: arg}
        if needs is None:
            for mname, fn in metrics.items():
                arg_names = self._infer_args(fn)  # may raise by design
                out[mname] = {arg: arg for arg in arg_names}
            return out

        # Case B: per-metric provided (values can be list OR dict) -> use as-is
        if isinstance(needs, dict) and any(
            isinstance(v, (dict, list)) for v in needs.values()
        ):
            for mname in metrics:
                if mname in needs:
                    spec = needs[mname]
                    if isinstance(spec, list):
                        out[mname] = list(spec)  # positional spec (as-is)
                    elif isinstance(spec, dict):
                        out[mname] = dict(spec)  # keyword spec (as-is)
                    else:
                        raise TypeError(
                            f"`needs[{mname}]` must be list or dict."
                        )
                else:
                    arg_names = self._infer_args(metrics[mname])
                    out[mname] = {arg: arg for arg in arg_names}
            return out

        # Case C: global positional list -> copy to every metric
        if isinstance(needs, list):
            for mname in metrics:
                out[mname] = list(needs)
            return out

        # Case D: global keyword mapping -> copy to every metric
        if isinstance(needs, dict):
            for mname in metrics:
                out[mname] = dict(needs)
            return out

        # Invalid structure
        raise TypeError(
            "`needs` must be one of:\n"
            "  - None\n"
            "  - list[(str|Callable), ...]             # positional (global)\n"
            "  - dict[arg -> (str|Callable)]            # keyword (global)\n"
            "  - dict[metric -> list[(str|Callable), ...]]    # metric-wise \n"
            "  - dict[metric -> dict[arg -> (str|Callable)]]  # metric-wise"
        )

    def _is_torchmetric(self, metric: Callable) -> bool:
        # Light check, we avoid importing torchmetrics here
        return (
            hasattr(metric, "update")
            and hasattr(metric, "compute")
            and hasattr(metric, "reset")
        )

    @torch.no_grad()
    def gather(self, trainer, obj):
        strat = trainer.strategy

        def _g(t):
            if not torch.is_tensor(t):
                return t
            t = t.detach()
            if t.device.type == "cpu":
                device_for_ddp = trainer.lightning_module.device
                t = t.to(device_for_ddp)
            g = strat.all_gather(t, sync_grads=False)
            g = g.reshape(-1, *t.shape[1:]) if g.dim() == t.dim() + 1 else g
            # Move back to CPU if user requested compute_on_cpu
            return g.cpu() if self.compute_on_cpu else g

        if isinstance(obj, torch.Tensor):
            return _g(obj)
        if isinstance(obj, list):
            return [_g(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _g(v) for k, v in obj.items()}
        else:
            raise TypeError("obj must be list or dict of tensors")
