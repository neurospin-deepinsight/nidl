##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import math
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from nidl.backbones.volume import VisionTransformer3D
from nidl.estimators.ssl import DINO, DINOv2
from nidl.utils import print_multicolor

EMBED_DIM = 16


def _tiny_vit(**kwargs):
    """ViT built for 32^3 global crops (2x2x2 patches of 16^3), accepting
    16^3 local crops (1 patch)."""
    params = {"img_size": 32, "patch_size": 16, "embed_dim": EMBED_DIM,
              "depth": 1, "num_heads": 2, "init_values": 1e-5,
              "dynamic_img_size": True, "mask_token": True}
    params.update(kwargs)
    return VisionTransformer3D(**params)


class MultiCropDataset(Dataset):
    """2 global crops (32^3) + 2 local crops (16^3) of random volumes."""

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return ([torch.rand(1, 32, 32, 32) for _ in range(2)]
                + [torch.rand(1, 16, 16, 16) for _ in range(2)])


class VolumeDataset(Dataset):
    """Plain 32^3 volumes, for `transform`."""

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.rand(1, 32, 32, 32)


class TestDINOv2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.loader = DataLoader(MultiCropDataset(), batch_size=4)

    def _make_model(self, **kwargs):
        params = {"encoder": _tiny_vit(), "proj_input_dim": EMBED_DIM,
                  "proj_hidden_dim": 32, "proj_bottleneck_dim": 8,
                  "proj_output_dim": 64, "proj_batch_norm": False,
                  "num_local_crops": 2, "freeze_last_layer": 1,
                  "lr_scheduler": "none", "max_epochs": 1,
                  "enable_checkpointing": False, "logger": False,
                  "random_state": 42}
        params.update(kwargs)
        return DINOv2(**params)

    def test_requires_mask_token(self):
        with self.assertRaises(ValueError):
            self._make_model(encoder=_tiny_vit(mask_token=False))

    def test_init_ibot_heads(self):
        model = self._make_model()
        for ps, pt in zip(model.student_ibot_head.parameters(),
                          model.teacher_ibot_head.parameters()):
            self.assertTrue(torch.equal(ps, pt))
            self.assertFalse(pt.requires_grad)
        names = [g["name"] for g in model._optim_params()]
        self.assertEqual(names, ["backbone", "head", "ibot_head"])

    def test_sample_masks(self):
        """Exactly half of the crops are masked, with ratios in range."""
        model = self._make_model(mask_ratio=(0.2, 0.75))
        masks = model._sample_masks(8, 100, torch.device("cpu"))
        n_hidden = masks.sum(dim=1)
        self.assertEqual(masks.shape, (8, 100))
        self.assertEqual(int((n_hidden > 0).sum()), 4)
        self.assertTrue(((n_hidden == 0) | (n_hidden >= 20)).all())
        self.assertTrue((n_hidden <= 75).all())
        # Tiny grid: a masked crop still hides at least one patch
        masks = model._sample_masks(4, 2, torch.device("cpu"))
        self.assertEqual(int((masks.sum(dim=1) > 0).sum()), 2)

    def test_shared_step(self):
        model = self._make_model()
        out = model._shared_step(next(iter(self.loader)))
        for name in ("loss", "dino", "ibot", "koleo"):
            self.assertTrue(torch.isfinite(out[name]), name)
        self.assertGreater(out["ibot"].item(), 0)
        self.assertEqual(out["z_student"].shape, (4, 4, 64))
        self.assertEqual(out["z_teacher"].shape, (2, 4, 64))
        out["loss"].backward()
        self.assertIsNotNone(model.student.mask_token.grad)
        self.assertIsNotNone(model.student_ibot_head.last_layer.weight_v.grad)

    def test_fit_transform(self):
        print(f"[{print_multicolor('DINOv2', display=False)}]...")
        model = self._make_model()
        teacher = [p.clone() for p in model.teacher_ibot_head.parameters()]
        model.fit(self.loader, val_dataloader=self.loader)
        for name in ("loss", "dino", "ibot", "koleo"):
            for stage in ("train", "val"):
                value = model.trainer.callback_metrics[f"{name}/{stage}"]
                self.assertTrue(torch.isfinite(value), f"{name}/{stage}")
        # The teacher iBOT head follows the student one (EMA), without grads
        changed = [not torch.equal(p0, p.cpu()) for p0, p in
                   zip(teacher, model.teacher_ibot_head.parameters())]
        self.assertTrue(any(changed))
        for p in model.teacher_ibot_head.parameters():
            self.assertIsNone(p.grad)
        z = model.transform(DataLoader(VolumeDataset(), batch_size=3))
        self.assertEqual(z.shape, (6, EMBED_DIM))

    def test_layerwise_lr(self):
        """Block i of a depth-L ViT: lr * decay**(L - i); embeddings:
        decay**(L + 1), patch embedding also x patch_embed_lr_mult."""
        model = self._make_model(encoder=_tiny_vit(depth=2),
                                 learning_rate=1.0, layerwise_lr_decay=0.5,
                                 patch_embed_lr_mult=0.2)
        lr = {id(p): g.get("lr", 1.0) for g in model._optim_params()
              for p in g["params"]}
        expected = {"patch_embed.proj.weight": 0.5 ** 3 * 0.2,
                    "pos_embed": 0.5 ** 3, "mask_token": 0.5 ** 3,
                    "blocks.0.attn.qkv.weight": 0.5 ** 2,
                    "blocks.1.attn.qkv.weight": 0.5, "norm.weight": 1.0}
        params = dict(model.student.named_parameters())
        for name, value in expected.items():
            self.assertAlmostEqual(lr[id(params[name])], value, msg=name)
        for p in model.student_head.parameters():
            self.assertEqual(lr[id(p)], 1.0)
        n_params = sum(1 for m in (model.student, model.student_head,
                                   model.student_ibot_head)
                       for _ in m.parameters())
        n_grouped = sum(len(list(g["params"]))
                        for g in model._optim_params())
        self.assertEqual(n_grouped, n_params)  # each parameter once
        self.assertEqual(len(lr), n_params)

    def test_fit_dinov2_recipe(self):
        """Sinkhorn-Knopp, weight decay schedule and layer-wise lr train."""
        model = self._make_model(centering="sinkhorn_knopp",
                                 weight_decay=0.04,
                                 final_weight_decay=0.4,
                                 layerwise_lr_decay=0.9,
                                 patch_embed_lr_mult=0.2, max_epochs=2,
                                 lr_scheduler="warmup_cosine",
                                 lr_scheduler_kwargs={"warmup_epochs": 1})
        self.assertEqual(model.ibot_loss.centering, "sinkhorn_knopp")
        model.fit(self.loader)
        for name in ("loss", "dino", "ibot", "koleo"):
            value = model.trainer.callback_metrics[f"{name}/train"]
            self.assertTrue(torch.isfinite(value), name)
        # 4 steps: last step at progress 3/4 of the cosine
        wd = 0.4 + 0.5 * (0.04 - 0.4) * (1 + math.cos(math.pi * 3 / 4))
        decays = {g["weight_decay"]
                  for g in model.trainer.optimizers[0].param_groups}
        self.assertEqual(len(decays), 2)
        self.assertAlmostEqual(min(decays), 0.0)
        self.assertAlmostEqual(max(decays), wd, places=6)

    def test_clip_grad_heads(self):
        """Default: only the backbone is clipped; with `clip_grad_heads`,
        each head too (separately)."""
        def norm(module):
            return torch.stack([p.grad.norm() for p in module.parameters()
                                if p.grad is not None]).norm().item()

        batch = next(iter(self.loader))
        for heads in (False, True):
            torch.manual_seed(0)
            model = self._make_model(clip_grad=1e-3, clip_grad_heads=heads)
            (model._shared_step(batch)["loss"] * 1e3).backward()
            model.on_after_backward()
            self.assertLessEqual(norm(model.student), 1e-3 * 1.01)
            for head in (model.student_head, model.student_ibot_head):
                if heads:
                    self.assertLessEqual(norm(head), 1e-3 * 1.01)
                else:
                    self.assertGreater(norm(head), 1e-3 * 1.01)

    def test_layerwise_lr_needs_vit(self):
        with self.assertRaises(ValueError):
            DINO(encoder=torch.nn.Linear(4, 4), proj_input_dim=4,
                 layerwise_lr_decay=0.9)

    def test_final_weight_decay_needs_weight_decay(self):
        with self.assertRaises(ValueError):
            self._make_model(weight_decay=0.0, final_weight_decay=0.4)


if __name__ == "__main__":
    unittest.main()
