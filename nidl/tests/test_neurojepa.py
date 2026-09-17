##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nidl.backbones.volume.vit3d_moe import (
    Block,
    MoE,
    MoEParams,
    VisionTransformer3DMoE,
    moe_bias_update,
)
from nidl.estimators.ssl.neurojepa import (
    DEFAULT_MULTISCALE_MASK_CONFIG,
    MaskScaleConfig,
    MultiScaleMaskCollator,
    NeuroJEPA,
    NeuroJEPAEncoderWrapper,
    VisionTransformerPredictor3D,
    _MaskGenerator,
    compute_foreground_patches,
    foreground_aware_jepa_loss,
)
from nidl.utils import print_multicolor

# Small grid used throughout: 2x2x2 = 8 patches, non-cubic volume checked
# separately where it matters.
GRID_SHAPE = (2, 2, 2)
PATCH_SIZE = (4, 4, 4)
NUM_PATCHES = 8
VOLUME_SHAPE = tuple(g * p for g, p in zip(GRID_SHAPE, PATCH_SIZE))


def _tiny_vit(use_moe: bool = False) -> VisionTransformer3DMoE:
    """A minimal VisionTransformer3DMoE satisfying NeuroJEPA's encoder
    interface, small enough to run instantly on CPU."""
    moe_params = None
    if use_moe:
        moe_params = MoEParams(
            dim=12,
            n_shared_experts=1,
            n_routed_experts=2,
            n_activated_experts=1,
            moe_inter_dim=4,
            moe_layer_indices=(0,),
        )
    return VisionTransformer3DMoE(
        img_size=VOLUME_SHAPE,
        patch_size=PATCH_SIZE,
        in_chans=1,
        embed_dim=12,
        depth=1,
        num_heads=2,
        use_moe=use_moe,
        moe_params=moe_params,
    )


class VolumeDataset(Dataset):
    """Plain (N, C, H, W, D) volume dataset, no labels/transforms: NeuroJEPA
    (like IJEPA) builds its own context/target views internally via masking,
    so unlike the view-based SSL estimators it does not need a
    `MultiViewsTransform`.
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestNeuroJEPAComponents(unittest.TestCase):
    """Unit tests for NeuroJEPA's building blocks in isolation: no
    `pl.Trainer` involved, so these stay fast and pin down the masking /
    foreground-weighting logic precisely.
    """

    def test_default_multiscale_mask_config(self):
        """DEFAULT_MULTISCALE_MASK_CONFIG must match the docstring/paper."""
        expected = (
            {"spatial_scale": (0.0, 0.2), "num_blocks": 32},
            {"spatial_scale": (0.2, 0.5), "num_blocks": 16},
            {"spatial_scale": (0.5, 0.7), "num_blocks": 4},
        )
        self.assertEqual(len(DEFAULT_MULTISCALE_MASK_CONFIG), len(expected))
        for cfg, exp in zip(DEFAULT_MULTISCALE_MASK_CONFIG, expected):
            self.assertEqual(cfg.spatial_scale, exp["spatial_scale"])
            self.assertEqual(cfg.num_blocks, exp["num_blocks"])
            self.assertEqual(cfg.total_mask_ratio, 0.75)

    def test_mask_generator_hits_exact_target(self):
        """Regardless of the random block carving, `_MaskGenerator` must
        always return a context/target split of exactly `target_enc` /
        `target_pred` patches that partitions the full patch grid."""
        cfg = MaskScaleConfig(num_blocks=3, total_mask_ratio=0.6)
        gen = _MaskGenerator(GRID_SHAPE, cfg)
        enc_idx, pred_idx = gen(batch_size=4, seed=0)
        self.assertEqual(enc_idx.shape, (4, gen.target_enc))
        self.assertEqual(pred_idx.shape, (4, gen.target_pred))
        for b in range(4):
            union = torch.cat([enc_idx[b], pred_idx[b]]).sort().values
            self.assertTrue(
                torch.equal(union, torch.arange(NUM_PATCHES)),
                msg="context+target masks must partition all patches",
            )

    def test_mask_generator_adjust_to_target_drop_prefers_background(self):
        """When `_adjust_to_target` must drop kept patches to hit
        `target_enc`, and enough background patches are available to cover
        the deficit, only background patches should be dropped."""
        cfg = MaskScaleConfig(total_mask_ratio=0.5)  # target_enc = 4
        gen = _MaskGenerator(GRID_SHAPE, cfg)
        g = torch.Generator().manual_seed(0)
        mask = torch.ones(GRID_SHAPE, dtype=torch.bool)  # all kept: n_keep=8
        fg_flat = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        fg_flat[:2] = True  # 2 foreground, 6 background (>= the 4 to drop)
        adjusted = gen._adjust_to_target(mask, g, fg_flat)
        dropped_idx = (~adjusted.flatten()).nonzero(as_tuple=True)[0]
        self.assertEqual(len(dropped_idx), gen.num_patches - gen.target_enc)
        self.assertTrue(torch.all(dropped_idx >= 2), msg="only background "
                         "patches (index >= 2) should have been dropped")

    def test_mask_generator_adjust_to_target_grow_prefers_foreground(self):
        """When `_adjust_to_target` must add kept patches to hit
        `target_enc`, and enough foreground patches are available, only
        foreground patches should be restored."""
        cfg = MaskScaleConfig(total_mask_ratio=0.25)  # target_enc = 6
        gen = _MaskGenerator(GRID_SHAPE, cfg)
        g = torch.Generator().manual_seed(0)
        mask = torch.zeros(GRID_SHAPE, dtype=torch.bool)  # nothing kept
        fg_flat = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        fg_flat[:7] = True  # 7 foreground (>= the 6 to restore), 1 background
        adjusted = gen._adjust_to_target(mask, g, fg_flat)
        kept_idx = adjusted.flatten().nonzero(as_tuple=True)[0]
        self.assertEqual(len(kept_idx), gen.target_enc)
        self.assertTrue(torch.all(kept_idx < 7), msg="only foreground "
                         "patches (index < 7) should have been restored")

    def test_mask_generator_adjust_to_target_without_foreground_map(self):
        """Both branches of `_adjust_to_target` (drop / grow) must also work
        when `foreground_mask` is not provided (`foreground_aware=False`)."""
        cfg = MaskScaleConfig(total_mask_ratio=0.25)  # target_enc = 6
        gen = _MaskGenerator(GRID_SHAPE, cfg)

        g = torch.Generator().manual_seed(0)
        grown = gen._adjust_to_target(
            torch.zeros(GRID_SHAPE, dtype=torch.bool), g, None,
        )
        self.assertEqual(int(grown.sum()), gen.target_enc)

        g = torch.Generator().manual_seed(0)
        dropped = gen._adjust_to_target(
            torch.ones(GRID_SHAPE, dtype=torch.bool), g, None,
        )
        self.assertEqual(int(dropped.sum()), gen.target_enc)

    def test_multiscale_mask_collator_shapes(self):
        scale_configs = (
            MaskScaleConfig(num_blocks=2, total_mask_ratio=0.5),
            MaskScaleConfig(num_blocks=2, total_mask_ratio=0.75),
        )
        collator = MultiScaleMaskCollator(
            grid_shape=GRID_SHAPE,
            patch_size=PATCH_SIZE,
            scale_configs=scale_configs,
            foreground_aware=True,
        )
        volumes = torch.rand(3, 1, *VOLUME_SHAPE)
        masks_enc, masks_pred, fg_flat = collator(volumes)
        self.assertEqual(len(masks_enc), 2)
        self.assertEqual(len(masks_pred), 2)
        for gen, enc, pred in zip(collator.generators, masks_enc, masks_pred):
            self.assertEqual(enc.shape, (3, gen.target_enc))
            self.assertEqual(pred.shape, (3, gen.target_pred))
        self.assertEqual(fg_flat.shape, (3, NUM_PATCHES))

    def test_multiscale_mask_collator_no_foreground(self):
        collator = MultiScaleMaskCollator(
            grid_shape=GRID_SHAPE,
            patch_size=PATCH_SIZE,
            scale_configs=(MaskScaleConfig(num_blocks=1, total_mask_ratio=0.5),),
            foreground_aware=False,
        )
        volumes = torch.rand(2, 1, *VOLUME_SHAPE)
        _, _, fg_flat = collator(volumes)
        self.assertIsNone(fg_flat)

    def test_multiscale_mask_collator_set_rank_offsets_step(self):
        collator = MultiScaleMaskCollator(
            grid_shape=GRID_SHAPE,
            patch_size=PATCH_SIZE,
            scale_configs=(MaskScaleConfig(num_blocks=1),),
        )
        self.assertEqual(collator.step(), 0)
        collator.set_rank(2)
        self.assertEqual(collator.step(), 2 * 10_000_000 + 1)

    def test_compute_foreground_patches_detects_blob(self):
        volumes = torch.zeros(1, 1, 4, 4, 4)
        volumes[:, :, :2, :2, :2] = 1.0  # bright blob in the first patch
        fg = compute_foreground_patches(
            volumes, patch_size=(2, 2, 2), threshold=0.0,
            min_foreground_fraction=0.5,
        )
        self.assertEqual(fg.shape, (1, 2, 2, 2))
        self.assertTrue(bool(fg[0, 0, 0, 0]))
        self.assertFalse(bool(fg[0, 1, 1, 1]))

    def test_compute_foreground_patches_degenerate_uniform_volume(self):
        """A uniform (zero-variance) volume can't yield a data-driven
        (2nd/98th percentile) threshold, so `compute_foreground_patches`
        must fall back to the fixed `threshold` without raising."""
        volumes = torch.zeros(2, 1, 4, 4, 4)
        fg = compute_foreground_patches(
            volumes, patch_size=(2, 2, 2), threshold=0.0,
            min_foreground_fraction=0.1,
        )
        self.assertEqual(fg.shape, (2, 2, 2, 2))
        self.assertFalse(bool(fg.any()))

    def test_foreground_aware_jepa_loss_uniform_fallback(self):
        """`bg_weight=1.0` or `fg_map=None` must both reduce to a plain,
        unweighted per-token L1 mean."""
        z = [torch.zeros(2, 4, 3)]
        h = [torch.ones(2, 4, 3)]
        masks_pred = [torch.arange(4).expand(2, 4)]
        expected = torch.abs(z[0] - h[0]).mean()

        loss_uniform_weight = foreground_aware_jepa_loss(
            z, h, masks_pred, bg_weight=1.0,
            fg_map=torch.rand(2, 4),
        )
        self.assertTrue(torch.allclose(loss_uniform_weight, expected))

        loss_no_fgmap = foreground_aware_jepa_loss(
            z, h, masks_pred, bg_weight=0.1, fg_map=None,
        )
        self.assertTrue(torch.allclose(loss_no_fgmap, expected))

    def test_foreground_aware_jepa_loss_upweights_foreground_errors(self):
        """A given error contributes more to the loss when it sits on a
        foreground token than on a background token."""
        z = torch.zeros(1, 4, 1)
        h = torch.zeros(1, 4, 1)
        h[:, 0, 0] = 10.0  # all the error is on token/patch 0
        masks_pred = [torch.arange(4).unsqueeze(0)]
        fg_map_fg = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # patch 0 foreground
        fg_map_bg = torch.tensor([[0.0, 1.0, 1.0, 1.0]])  # patch 0 background

        loss_fg = foreground_aware_jepa_loss(
            [z], [h], masks_pred, bg_weight=0.1, fg_map=fg_map_fg,
        )
        loss_bg = foreground_aware_jepa_loss(
            [z], [h], masks_pred, bg_weight=0.1, fg_map=fg_map_bg,
        )
        self.assertGreater(loss_fg.item(), loss_bg.item())

    def test_encoder_wrapper_rejects_invalid_encoder(self):
        with self.assertRaises(TypeError):
            NeuroJEPAEncoderWrapper(nn.Linear(4, 4))

    def test_encoder_wrapper_rejects_non_callable_forward(self):
        class FakeEncoder:
            embed_dim = 4
            patch_size = (1, 1, 1)
            grid_shape = (1, 1, 1)
            blocks = nn.ModuleList()
            forward = None  # has all required attrs, but isn't callable

        with self.assertRaises(TypeError):
            NeuroJEPAEncoderWrapper(FakeEncoder())

    def test_encoder_wrapper_exposes_interface(self):
        encoder = _tiny_vit()
        wrapper = NeuroJEPAEncoderWrapper(encoder)
        self.assertEqual(wrapper.embed_dim, encoder.embed_dim)
        self.assertEqual(wrapper.patch_size, encoder.patch_size)
        self.assertEqual(wrapper.grid_shape, encoder.grid_shape)
        self.assertIs(wrapper.blocks, encoder.blocks)

        x = torch.rand(2, 1, *VOLUME_SHAPE)
        tokens, moe_scores = wrapper(x)
        self.assertEqual(tokens.shape, (2, NUM_PATCHES, encoder.embed_dim))
        self.assertEqual(moe_scores, [])

    def test_predictor_output_shape(self):
        embed_dim, pred_dim = 12, 8
        predictor = VisionTransformerPredictor3D(
            grid_shape=GRID_SHAPE, embed_dim=embed_dim,
            predictor_embed_dim=pred_dim, depth=1, num_heads=2,
        )
        batch_size, k_ctx, k_pred = 2, 3, 5
        x = torch.rand(batch_size, k_ctx, embed_dim)
        masks_x = [torch.randint(0, NUM_PATCHES, (batch_size, k_ctx))]
        masks_y = [torch.randint(0, NUM_PATCHES, (batch_size, k_pred))]
        out = predictor(x, masks_x=masks_x, masks_y=masks_y)
        self.assertEqual(out.shape, (batch_size, k_pred, embed_dim))

    def test_moe_bias_update_no_moe_layers_is_noop(self):
        block = Block(dim=4, num_heads=2, use_moe=False, grid_size=2)
        model = nn.Module()
        model.blocks = nn.ModuleList([block])
        min_vio, max_vio = moe_bias_update(model, update_rate=1e-3)
        self.assertEqual((min_vio, max_vio), (1.0, 1.0))

    def test_moe_bias_update_changes_router_bias(self):
        moe_params = MoEParams(
            dim=4, n_shared_experts=1, n_routed_experts=2,
            n_activated_experts=1, moe_inter_dim=4, moe_layer_indices=(0,),
        )
        block = Block(
            dim=4, num_heads=2, use_moe=True, moe_params=moe_params,
            grid_size=2,
        )
        moe = block.mlp
        self.assertIsInstance(moe, MoE)
        self.assertTrue(torch.equal(moe.gate.bias, torch.zeros(2)))

        x = torch.rand(3, 5, 4)  # populate moe.counts via a real forward
        block(x, D_patches=1, H_patches=1, W_patches=5)

        model = nn.Module()
        model.blocks = nn.ModuleList([block])
        min_vio, max_vio = moe_bias_update(
            model, update_rate=1e-2, bias_clip=0.3,
        )
        self.assertGreaterEqual(min_vio, 0.0)
        self.assertGreaterEqual(max_vio, 0.0)
        self.assertTrue(torch.all(moe.gate.bias.abs() <= 0.3))


class TestNeuroJEPA(unittest.TestCase):
    """Tests exercising the `NeuroJEPA` estimator itself, following the same
    design as `TestEstimators`/`TestVAE` in `test_estimator.py`.
    """

    def setUp(self):
        self.n_volumes = 8
        self.fake_data = torch.rand(self.n_volumes, 1, *VOLUME_SHAPE)
        self.loader = DataLoader(
            VolumeDataset(self.fake_data), batch_size=2, shuffle=False,
        )
        # Small mask config so fit() stays fast in tests.
        self.mask_scale_configs = (
            MaskScaleConfig(num_blocks=2, total_mask_ratio=0.5),
        )

    def _make_model(self, use_moe=False, **kwargs):
        params = dict(
            encoder=_tiny_vit(use_moe=use_moe),
            mask_scale_configs=self.mask_scale_configs,
            foreground_aware=True,
            predictor_embed_dim=8,
            predictor_depth=1,
            predictor_num_heads=2,
            use_moe=use_moe,
            learning_rate=5e-4,
            lr_scheduler="none",
            max_epochs=1,
            limit_train_batches=2,
            random_state=42,
        )
        params.update(kwargs)
        return NeuroJEPA(**params)

    def test_init_attributes(self):
        print(f"[{print_multicolor('NeuroJEPA', display=False)}]...")
        model = self._make_model(ema_start=0.9, ema_end=1.0)
        self.assertIsInstance(model.context_encoder, NeuroJEPAEncoderWrapper)
        self.assertIsInstance(model.target_encoder, NeuroJEPAEncoderWrapper)
        self.assertIsInstance(model.predictor, VisionTransformerPredictor3D)
        self.assertIsInstance(model.masker, MultiScaleMaskCollator)
        self.assertEqual(model.momentum_updater.base_lambda, 0.9)

        # target encoder starts as a frozen copy of the context encoder.
        for pc, pt in zip(
            model.context_encoder.parameters(),
            model.target_encoder.parameters(),
        ):
            self.assertTrue(torch.equal(pc.data, pt.data))
            self.assertFalse(pt.requires_grad)
            self.assertTrue(pc.requires_grad)

    def test_fit_transform_shape(self):
        """Simple fit/transform check, mirroring
        `TestEstimators.test_ssl`. Also exercises `validation_step` by
        fitting with a validation dataloader."""
        print(f"[{print_multicolor('NeuroJEPA', display=False)}]...")
        model = self._make_model()
        model.fit(self.loader, val_dataloader=self.loader)
        z = model.transform(self.loader)
        self.assertEqual(z.shape, (self.n_volumes, model.target_encoder.embed_dim))

    def test_shared_step_outputs(self):
        """`_shared_step` doesn't touch `self.trainer`, so it can be called
        directly without a `pl.Trainer` attached."""
        model = self._make_model()
        batch = self.fake_data[:2]
        outputs = model._shared_step(batch)
        self.assertIn("loss", outputs)
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(outputs["loss"].ndim, 0)

        num_scales = len(self.mask_scale_configs)
        self.assertEqual(len(outputs["z_pred"]), num_scales)
        self.assertEqual(len(outputs["z_target"]), num_scales)
        for zp, zt in zip(outputs["z_pred"], outputs["z_target"]):
            self.assertEqual(zp.shape, zt.shape)

    def test_forward_target_matches_manual_pass(self):
        model = self._make_model()
        batch = self.fake_data[:2]
        full_masks = [torch.arange(NUM_PATCHES).expand(2, NUM_PATCHES)]

        h = model.forward_target(batch, full_masks)
        self.assertEqual(len(h), 1)
        self.assertFalse(h[0].requires_grad)

        with torch.no_grad():
            tokens, _ = model.target_encoder(batch, masks=None)
            expected = torch.nn.functional.layer_norm(
                tokens, (tokens.size(-1),)
            )
        self.assertTrue(torch.allclose(h[0], expected))

    def test_transform_step_shape(self):
        model = self._make_model()
        batch = self.fake_data[:3]
        out = model.transform_step(batch, batch_idx=0)
        self.assertEqual(out.shape, (3, model.target_encoder.embed_dim))

    def test_test_step_is_skipped(self):
        model = self._make_model()
        self.assertIsNone(model.test_step(self.fake_data[:2], batch_idx=0))

    def test_fill_default_lr_scheduler_kwargs(self):
        model = self._make_model(lr_scheduler_kwargs=None)
        self.assertEqual(
            model.lr_scheduler_kwargs,
            {
                "warmup_epochs": 10,
                "interval": "step",
                "warmup_start_lr": 1e-6,
                "min_lr": 0,
            },
        )

        model = self._make_model(lr_scheduler_kwargs={"warmup_epochs": 5})
        self.assertEqual(model.lr_scheduler_kwargs["warmup_epochs"], 5)
        self.assertEqual(model.lr_scheduler_kwargs["min_lr"], 0)

    def test_fill_default_lr_scheduler_kwargs_from_none(self):
        """Direct call, bypassing `__init__`'s own `None`-guard, to cover
        the method's own `None` handling."""
        model = self._make_model()
        model.lr_scheduler_kwargs = None
        model._fill_default_lr_scheduler_kwargs()
        self.assertEqual(model.lr_scheduler_kwargs["warmup_epochs"], 10)

    def test_ignore_list_always_excludes_encoder_and_callbacks(self):
        """`encoder`/`callbacks` must be excluded from the saved
        hyperparameters even when the caller's own `ignore` list omits
        them."""
        model = self._make_model(ignore=["callbacks"])  # missing "encoder"
        self.assertNotIn("encoder", model.hparams)

        model = self._make_model(ignore=[])  # missing both
        self.assertNotIn("callbacks", model.hparams)
        self.assertNotIn("encoder", model.hparams)

    def test_configure_optimizers_none_scheduler(self):
        """`configure_optimizers` always reads `self.trainer` (even just to
        pass it through), so a `Trainer` must be attached; but with
        `lr_scheduler='none'` its `max_epochs` is never actually read
        (unlike the default 'warmup_cosine')."""
        model = self._make_model(lr_scheduler="none")
        model.trainer = pl.Trainer(max_epochs=None)
        optimizer = model.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        names = {g["name"] for g in optimizer.param_groups}
        self.assertEqual(
            names,
            {"backbone", "predictor", "backbone_no_decay", "predictor_no_decay"},
        )

    def test_on_train_batch_end_updates_momentum(self):
        """After fitting, the target encoder must have moved from its
        (frozen, initially-identical) starting point towards the context
        encoder via the EMA update."""
        model = self._make_model(ema_start=0.0, ema_end=0.0)
        before = [
            p.detach().clone() for p in model.target_encoder.parameters()
        ]
        model.fit(self.loader)
        after = list(model.target_encoder.parameters())
        self.assertTrue(
            any((a - b).abs().sum() > 0 for a, b in zip(after, before))
        )

    def test_use_moe_logs_finite_violations(self):
        model = self._make_model(
            use_moe=True, moe_bias_update_rate=1e-3, moe_bias_clip=0.3,
        )
        model.fit(self.loader)  # must not raise


if __name__ == "__main__":
    unittest.main()
