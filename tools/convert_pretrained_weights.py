##########################################################################
# NSAp - Copyright (C) CEA, 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Convert published checkpoints to the nidl format.

This script produced the repositories of the `nidl-neurospin
<https://huggingface.co/nidl-neurospin>`_ organization:

- for the weights that can be published again, it renames the original
  checkpoint after the modules of the nidl estimator, loads it strictly and
  saves the estimator with
  :meth:`~nidl.estimators.base.NidlHubMixin.save_pretrained`, which writes the
  `config.json` and `model.safetensors` to upload;
- for the weights that cannot, it only writes the `config.json`, whose
  ``"weights"`` entry points to the repository of the authors (see
  :mod:`nidl.estimators.external`).

Run it with::

    python convert_pretrained_weights.py brainiac-vit-b out/ \
        --source BrainIAC.ckpt
    python convert_pretrained_weights.py neuro-jepa-vit-b out/

Sources of the original checkpoints:

- yaware-densenet121-bhb10k:
  https://github.com/Duplums/yAwareContrastiveLearning
- neuro-simclr-resnet18: https://github.com/emilykaczmarek/3D-Neuro-SimCLR
- brainiac-vit-b: https://github.com/AIM-KannLab/BrainIAC
- neuro-jepa-vit-b and neuro-jepa-vit-l: https://github.com/NYUMedML/Neuro-JEPA

The model names are the names of the repositories.
"""

import argparse
import json
import re
from pathlib import Path

import torch

from nidl.estimators.ssl import SimCLR, YAwareContrastiveLearning
from nidl.info import __version__

MODELS = {
    "yaware-densenet121-bhb10k": {
        "estimator": YAwareContrastiveLearning,
        "checkpoint_key": "model",
        "key_mapping": (
            (r"^module\.hidden_representation\.", "projection_head.layers.0."),
            (r"^module\.head_projection\.", "projection_head.layers.2."),
            (r"^module\.features\.", "encoder.features."),
        ),
        "config": {
            "estimator_kwargs": {
                "proj_input_dim": 1024,
                "proj_hidden_dim": 512,
                "proj_output_dim": 128,
                "bandwidth": 25.0,
                "temperature": 0.1,
                "learning_rate": 1e-4,
                "weight_decay": 5e-5,
            },
            "encoder": {
                "name": "densenet121",
                "kwargs": {"n_embedding": None},
            },
            "input": {"shape": [1, 121, 145, 121], "range": None},
            "preprocessing": {"transform": "ZNormalization", "kwargs": {}},
        },
    },
    "neuro-simclr-resnet18": {
        "estimator": SimCLR,
        "checkpoint_key": "model_state_dict",
        "key_mapping": (
            (r"^module\.encoder\.", "encoder."),
            (r"^module\.projector\.", "projection_head.layers."),
        ),
        "config": {
            "estimator_kwargs": {
                "proj_input_dim": 512,
                "proj_hidden_dim": 512,
                "proj_output_dim": 64,
                "proj_bias": False,
                "temperature": 0.5,
            },
            "encoder": {
                "name": "resnet18",
                "kwargs": {
                    "n_embedding": None,
                    "initial_stride": 1,
                    "downsample_bias": True,
                },
            },
            "input": {"shape": [1, 150, 192, 192], "range": None},
            "preprocessing": {
                "transform": "NeuroSimCLRPreprocessing",
                "kwargs": {},
            },
        },
    },
    "brainiac-vit-b": {
        "estimator": SimCLR,
        "checkpoint_key": "state_dict",
        "key_mapping": (
            (
                r"^backbone\.patch_embedding\.position_embeddings",
                "encoder.pos_embed",
            ),
            (
                r"^backbone\.patch_embedding\.patch_embeddings",
                "encoder.patch_embed.proj",
            ),
            (r"^backbone\.", "encoder."),
            (r"\.mlp\.linear1\.", ".mlp.fc1."),
            (r"\.mlp\.linear2\.", ".mlp.fc2."),
            (r"\.attn\.out_proj\.", ".attn.proj."),
        ),
        "config": {
            "estimator_kwargs": {
                "proj_input_dim": 768,
                "proj_hidden_dim": 768,
                "proj_output_dim": 2048,
                "proj_batch_norm": True,
                "temperature": 0.1,
            },
            # The original ViT has no class token and its position embedding
            # only covers patch tokens. Its feature is the first patch token,
            # replaced by their mean, after the original final normalization.
            "encoder": {
                "name": "VisionTransformer3D",
                "kwargs": {
                    "img_size": 96,
                    "patch_size": 16,
                    "class_token": False,
                    "global_pool": "avg",
                    "fc_norm": False,
                    "no_embed_class": True,
                    "qkv_bias": False,
                },
            },
            "input": {"shape": [1, 96, 96, 96], "range": None},
            "preprocessing": {
                "transform": "BrainIACPreprocessing",
                "kwargs": {},
            },
        },
    },
}



def _neuro_jepa(repo_id, revision, embed_dim, depth, num_heads, inter_dim):
    return {
        "estimator_kwargs": {
            "use_moe": True,
            "predictor_embed_dim": 384,
            "predictor_depth": 12,
            "predictor_num_heads": 12,
            "foreground_aware": True,
            "moe_bias_clip": 0.3,
            "moe_bias_update_rate": 0.0001,
        },
        "encoder": {
            "name": "VisionTransformer3DMoE",
            "kwargs": {
                "img_size": [96, 108, 96],
                "patch_size": [12, 12, 12],
                "in_chans": 1,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "use_moe": True,
                "moe_params": {
                    "dim": embed_dim,
                    "n_shared_experts": 2,
                    "n_routed_experts": 16,
                    "n_activated_experts": 6,
                    "moe_inter_dim": inter_dim,
                    "score_func": "softmax",
                    "route_scale": 4.0,
                    "bias_clip": 0.3,
                    "bias_update_rate": 0.0001,
                    "moe_layer_indices": list(range(1, depth, 2)),
                },
                "attn_gate": True,
                "norm_eps": 1e-06,
            },
        },
        "input": {"shape": [1, 96, 108, 96], "range": [0.0, 1.0]},
        "preprocessing": {"transform": "NeuroJEPAPreprocessing", "kwargs": {}},
        "weights": {
            "repo_id": repo_id,
            "revision": revision,
            "filename": "model.safetensors",
            "converter": "NeuroJEPAWeights",
            "missing_ok": ["predictor."],
        },
    }


#: Weights loaded from the repository of their authors (CC-BY-NC-ND 4.0).
EXTERNAL_MODELS = {
    "neuro-jepa-vit-b": {
        "estimator": "NeuroJEPA",
        "config": _neuro_jepa(
            "NYUMedML/Neuro-JEPA",
            "f55b154091aa7ac5ca23632918e216cd92d9f2cb",
            768, 12, 12, 384,
        ),
    },
    "neuro-jepa-vit-l": {
        "estimator": "NeuroJEPA",
        "config": _neuro_jepa(
            "NYUMedML/Neuro-JEPA-Large",
            "61b8c9cc8c1b9ade335a10844994cfb903e2549f",
            1024, 24, 16, 512,
        ),
    },
}


def load_checkpoint(path, key):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)[key]


def convert(name, source, output):
    model_info = MODELS[name]
    config = {
        "nidl_version": __version__,
        "estimator": model_info["estimator"].__name__,
        **model_info["config"],
    }
    state_dict = {}
    checkpoint = load_checkpoint(source, model_info["checkpoint_key"])
    for key, value in checkpoint.items():
        for pattern, replacement in model_info["key_mapping"]:
            key = re.sub(pattern, replacement, key)
        if key in state_dict:
            raise ValueError(f"Several weights are renamed to '{key}'.")
        state_dict[key] = value
    model = model_info["estimator"].from_config(config, accelerator="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.save_pretrained(output)


def write_external_config(name, output):
    model_info = EXTERNAL_MODELS[name]
    config = {
        "nidl_version": __version__,
        "estimator": model_info["estimator"],
        **model_info["config"],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("model", choices=sorted({*MODELS, *EXTERNAL_MODELS}))
    parser.add_argument("output", help="folder to write")
    parser.add_argument("--source", help="original checkpoint")
    args = parser.parse_args()
    if args.model in EXTERNAL_MODELS:
        output = write_external_config(args.model, args.output)
    elif args.source is None:
        parser.error(f"'{args.model}' needs the original checkpoint.")
    else:
        output = convert(args.model, args.source, args.output)
    print(f"Saved to {output}")
