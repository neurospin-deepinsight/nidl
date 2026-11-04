##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Render the registry of pretrained models (`pretrained_models.json`).

Usage in a page::

    .. pretrained-models::
       :family: published
       :view: table

with `view` among "table" (summary), "cards" (one dropdown per model) and
"openbhb" (brain age on OpenBHB).
"""

import json
from pathlib import Path
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

HUB = "https://huggingface.co"
ESTIMATORS = "nidl.estimators.ssl"
PREPROCESSING = "nidl.transforms.volume.preprocessing"


def _label(model):
    return "pretrained-" + model["repo_id"].split("/")[-1]


def _python(value):
    """Format JSON values as Python literals, lists as tuples."""
    if isinstance(value, list):
        return "(" + ", ".join(_python(v) for v in value) + ")"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _call(prefix, arguments, width=72):
    """Format a call, one argument per line if it is too long."""
    line = f"{prefix}({', '.join(arguments)})"
    if len(line) <= width or not arguments:
        return [line]
    return [f"{prefix}(", *(f"    {a}," for a in arguments), ")"]


def _hub(repo_id):
    return f"`{repo_id} <{HUB}/{repo_id}>`__"


def _count(value):
    return "n/r" if value is None else value


def _table(models):
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :class: pretrained-models",
        "",
        "   * - Model",
        "     - Encoder",
        "     - Pretraining modalities",
        "     - Subjects",
        "     - Volumes",
        "     - Repository",
    ]
    for model in models:
        lock = " :fas:`lock`" if model["gated"] else ""
        repo_id = model["repo_id"]
        lines += [
            f"   * - :ref:`{model['name']} <{_label(model)}>`{lock}",
            (
                f"     - {model['encoder']} ({model['parameters']}, "
                f"{model['features']} features)"
            ),
            f"     - {model['modality']}",
            f"     - {_count(model['subjects'])}",
            f"     - {_count(model['volumes'])}",
            f"     - `{repo_id.split('/')[1]} <{HUB}/{repo_id}>`__",
        ]
    lines += [
        "",
        (
            "Subjects and volumes used for pretraining, as reported by the "
            "authors (n/r: not reported)."
        ),
    ]
    return lines


def _openbhb(models, info):
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Model",
        "     - OpenBHB modality",
        "     - MAE (years)",
        "     - R²",
    ]
    for model in sorted(models, key=lambda m: m["openbhb"]["mae"]):
        result = model["openbhb"]
        lines += [
            f"   * - :ref:`{model['name']} <{_label(model)}>`",
            f"     - ``{result['modality']}``",
            f"     - {result['mae']:.2f}",
            f"     - {result['r2']:.2f}",
        ]
    lines += [
        "",
        (
            f"{info['description']}, computed with nidl "
            f"{info['nidl_version']} on {info['date']}, see "
            f":ref:`{info['example']}`."
        ),
    ]
    return lines


def _cards(models):
    lines = []
    for model in models:
        openbhb = model["openbhb"]
        kwargs = [f"{k}={_python(v)}" for k, v in openbhb["kwargs"].items()]
        estimator = model["estimator"]
        paper = f"`{model['paper']['title']} <{model['paper']['url']}>`__"
        transform = f"{PREPROCESSING}.{model['preprocessing']}"
        lines += [
            f".. _{_label(model)}:",
            "",
            f".. dropdown:: {model['name']}",
            "   :class-title: sd-text-primary",
            "",
            f"   :Repository: {_hub(model['repo_id'])}",
            f"   :Estimator: :class:`~{ESTIMATORS}.{estimator}`",
            (
                f"   :Encoder: {model['encoder']}, {model['parameters']} "
                f"parameters, {model['features']}-dimensional features"
            ),
            f"   :Pretraining: {model['pretraining']}",
            f"   :Expected input: {model['input']}",
            f"   :Preprocessing: :class:`~{transform}`",
            f"   :Fidelity: {model['fidelity']}",
            f"   :License: {model['license']}",
            f"   :Paper: {paper}",
            f"   :Original code: {model['code']}",
        ]
        if model["gated"]:
            lines += [
                "",
                "   .. important::",
                "",
                "      The weights are gated: request access on",
                f"      {model['weights']}, then authenticate with",
                "      ``huggingface-cli login``.",
            ]
        lines += [
            "",
            "   .. code-block:: python",
            "",
            f"      from {ESTIMATORS} import {estimator}",
            "",
            *(
                f"      {line}"
                for line in _call(
                    f"model = {estimator}.from_pretrained",
                    [f'"{model["repo_id"]}"'],
                )
            ),
            f"      # Parameters for OpenBHB {openbhb['modality']} images",
            *(
                f"      {line}"
                for line in _call(
                    "preprocess = model.get_preprocessing", kwargs
                )
            ),
            "",
        ]
    return lines


class PretrainedModels(SphinxDirective):
    option_spec: ClassVar[dict] = {
        "family": directives.unchanged_required,
        "view": lambda arg: directives.choice(
            arg, ("table", "cards", "openbhb")
        ),
    }

    def run(self):
        path = Path(self.env.srcdir) / self.config.pretrained_models_registry
        self.env.note_dependency(str(path))
        registry = json.loads(path.read_text())
        models = [
            model
            for model in registry["models"]
            if model["family"] == self.options["family"]
        ]
        if not models:
            return []
        view = self.options.get("view", "table")
        if view == "table":
            lines = _table(models)
        elif view == "openbhb":
            lines = _openbhb(models, registry["openbhb"])
        else:
            lines = _cards(models)
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source=str(path)), self.content_offset, container
        )
        return container.children


def setup(app):
    app.add_config_value(
        "pretrained_models_registry", "pretrained_models.json", "env"
    )
    app.add_directive("pretrained-models", PretrainedModels)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
