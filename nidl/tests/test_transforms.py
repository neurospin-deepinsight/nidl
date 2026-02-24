##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from nidl.transforms import MultiViewsTransform, Identity
from nidl.volume.transforms.augmentation import (
    RandomRotation, RandomFlip, RandomResizedCrop, RandomErasing,
    RandomGaussianBlur, RandomGaussianNoise
)
from nidl.volume.transforms.preprocessing import (
    ZNormalization, RobustRescaling, CropOrPad, Resize, Resample   
)


class TestTransform(unittest.TestCase):

    def setUp(self):
        self.transforms_cls = [
            RandomRotation, RandomFlip, RandomErasing,
            RandomGaussianBlur, ZNormalization, RobustRescaling
            ] # all these transforms modify input
        self.numpy_volume = np.arange(2 * 4 * 4 * 4).reshape(2, 4, 4, 4)
        
    def test_probability_to_apply(self):
        for tf_cls in self.transforms_cls:
            tf = tf_cls(p=0.0) # never apply
            out = tf(self.numpy_volume)
            self.assertTrue(np.all(out == self.numpy_volume))
            tf = tf_cls(p=1.0) # always apply
            out = tf(self.numpy_volume)
            self.assertTrue(np.any(out != self.numpy_volume), f"Issue with transformation {tf}")
            for p in [-1, 3, [1, 2], "not number"]: # invalid
                with self.assertRaises(ValueError):
                    tf = tf_cls(p=p)
            
    def test_invalid_input_type(self):
        for tf_cls in self.transforms_cls:
            tf = tf_cls()
            with self.assertRaises(TypeError):
                tf([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]) 


class TestMultiViewsTransform(unittest.TestCase):
    def setUp(self):
        self.mock_transform = MagicMock(side_effect=lambda x: x + 1)
        self.input_data = np.array([1, 2, 3])

    def test_single_transform_multiple_views(self):
        mvt = MultiViewsTransform(self.mock_transform, n_views=3)
        outputs = mvt(self.input_data)
        self.assertEqual(len(outputs), 3)
        for out in outputs:
            np.testing.assert_array_equal(out, self.input_data + 1)

    def test_sequence_of_transforms(self):
        t1 = MagicMock(side_effect=lambda x: x * 2)
        t2 = MagicMock(side_effect=lambda x: x + 3)
        mvt = MultiViewsTransform([t1, t2])
        outputs = mvt(self.input_data)
        self.assertEqual(len(outputs), 2)
        np.testing.assert_array_equal(outputs[0], self.input_data * 2)
        np.testing.assert_array_equal(outputs[1], self.input_data + 3)

    def test_nviews(self):
        for n_views in [1, 2]:
            mvt = MultiViewsTransform(self.mock_transform, n_views=n_views)
            outputs = mvt(self.input_data)
            self.assertEqual(len(outputs), n_views)
            np.testing.assert_array_equal(outputs[0], self.input_data + 1)

    def test_invalid_transform_type(self):
        with self.assertRaises(TypeError):
            MultiViewsTransform("not_a_callable")

    def test_sequence_with_non_callable(self):
        with self.assertRaises(TypeError):
            MultiViewsTransform([self.mock_transform, "bad_transform"])

    def test_non_callable(self):
        with self.assertRaises(TypeError):
            MultiViewsTransform("bad transform")

    def test_invalid_nviews_type(self):
        with self.assertRaises(TypeError):
            MultiViewsTransform(self.mock_transform, n_views="two")

    def test_negative_nviews(self):
        with self.assertRaises(ValueError):
            MultiViewsTransform(self.mock_transform, n_views=-1)

    def test_sequence_with_nviews_not_one(self):
        with self.assertRaises(ValueError):
            MultiViewsTransform([self.mock_transform], n_views=2)


class TestIdentity(unittest.TestCase):
    def setUp(self):
        self.numpy_array = np.array([1, 2, 3])
        self.torch_tensor = torch.tensor([1, 2, 3])
    
    def test_numpy_input(self):
        tf = Identity()
        self.assertTrue(np.all(self.numpy_array == tf(self.numpy_array)))

    def test_torch_input(self):
        tf = Identity()
        self.assertTrue(torch.all(self.torch_tensor == tf(self.torch_tensor)))
    
    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            Identity()("abcd")


class TestRandomRotation(unittest.TestCase):
    def setUp(self):
        self.numpy_volume = np.arange(2 * 4 * 4 * 4).reshape(2, 4, 4, 4).astype(np.float32)
        self.torch_volume = torch.from_numpy(self.numpy_volume)
        self.identity_affine = np.eye(4)
        self.ras_affine = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def test_voxel_axes_numpy(self):
        for axes in [(1, 2), ((1, 2),), ((0, 1), (1, 2), (0, 2))]:
            for dtype in [np.int16, np.int32, np.int64, np.float16, 
                          np.float32, np.float64, np.uint8, np.uint16]:
                t = RandomRotation(axes=axes, rotation_probability=1.0)
                out = t(self.numpy_volume.astype(dtype))
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.shape, self.numpy_volume.shape)
                self.assertEqual(out.dtype, dtype)

    def test_voxel_axes_torch(self):
        for axes in [(1, 2), ((1, 2),), ((0, 1), (1, 2), (0, 2))]:
            for dtype in [torch.int16, torch.int32, torch.int64, 
                          torch.float16, torch.float32, torch.float64]:
                t = RandomRotation(axes=axes, rotation_probability=1.0)
                out = t(self.torch_volume.to(dtype))
                self.assertIsInstance(out, torch.Tensor)
                self.assertEqual(tuple(out.shape), tuple(self.torch_volume.shape))
                self.assertEqual(out.dtype, dtype)

    def test_anatomical_axes_with_affine(self):
        t = RandomRotation(axes=("LR", "AP"), rotation_probability=1.0)
        out = t(self.numpy_volume, affine=self.ras_affine)
        self.assertEqual(out.shape, self.numpy_volume.shape)

    def test_anatomical_axes_missing_affine(self):
        t = RandomRotation(axes=(("LR", "AP"),), rotation_probability=1.0)
        out = t(self.numpy_volume, affine=None)
        self.assertTrue(np.any(out != self.numpy_volume))

    def test_rotation_probability_zero(self):
        t = RandomRotation(axes=((1, 2),), rotation_probability=0.0)
        out = t(self.numpy_volume.copy())
        np.testing.assert_array_equal(out, self.numpy_volume)

    def test_invalid_axis_length(self):
        with self.assertRaises(ValueError):
            RandomRotation(axes=(("LR",),))

    def test_invalid_anatomical_label(self):
        with self.assertRaises(ValueError):
            t = RandomRotation(axes=(("XX", "IS"),))

    def test_consistency_numpy_torch(self):
        t = RandomRotation(axes=(("LR", "IS"),), rotation_probability=1.0)
        np_out = t(self.numpy_volume.copy(), affine=self.ras_affine)
        torch_out = t(self.torch_volume.clone(), affine=self.ras_affine)
        self.assertEqual(np_out.shape, self.numpy_volume.shape)
        self.assertEqual(tuple(torch_out.shape), tuple(self.torch_volume.shape))
    

class TestRandomFlip(unittest.TestCase):
    def setUp(self):
        self.volume_np = np.arange(2 * 4 * 4 * 4).reshape(2, 4, 4, 4)
        self.volume_torch = torch.from_numpy(self.volume_np.copy())
        self.ras_affine = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) # 'A', 'L', 'S' orientation

    def test_flip_np_axes(self):
        # Always flip
        transform = RandomFlip(axes=(0, 1, 2), flip_probability=1.0)
        flipped = transform(self.volume_np)
        # Manually flip with numpy
        expected = np.flip(self.volume_np, axis=(1, 2, 3))
        np.testing.assert_array_equal(flipped, expected)

    def test_flip_torch_axes(self):
        # Always flip
        transform = RandomFlip(axes=(0, 1, 2), flip_probability=1.0)
        flipped = transform(self.volume_torch)
        # Manually flip with torch
        expected = torch.flip(self.volume_torch, dims=(1, 2, 3))
        self.assertTrue(torch.equal(flipped, expected))

    def test_no_flip_probability_zero(self):
        transform = RandomFlip(axes=(0, 1, 2), flip_probability=0.0)
        flipped_np = transform(self.volume_np)
        flipped_torch = transform(self.volume_torch)

        np.testing.assert_array_equal(flipped_np, self.volume_np)
        self.assertTrue(torch.equal(flipped_torch, self.volume_torch))

    def test_anatomical_axes_with_affine(self):
        volume = np.random.rand(4, 10, 10, 10)
        transform = RandomFlip(axes=("AP", "IS"), flip_probability=1.0)
        flipped = transform(volume, affine=self.ras_affine)

        expected = np.flip(volume, axis=(1, 3)) # no LR flip
        self.assertTrue(np.allclose(flipped, expected))

    def test_invalid_axes_value(self):
        with self.assertRaises(ValueError):
            RandomFlip(axes=(3,))  # Invalid axis index

        with self.assertRaises(ValueError):
            RandomFlip(axes=("XY",))  # Invalid anatomical label

    def test_invalid_input(self):
        transform = RandomFlip(axes="LR", flip_probability=1.0)
        with self.assertRaises(ValueError):
            transform(np.arange(2 * 4 * 4 * 4).reshape(2, 4, 4, 4, 1))
        with self.assertRaises(ValueError):
            transform(np.arange(2 * 4).reshape(2, 4))
    
    def test_identity_affine_default(self):
        # Make sure identity affine works with default RAS assumptions
        volume = np.random.rand(4, 5, 5, 5)
        transform = RandomFlip(axes="LR", flip_probability=1.0)
        out = transform(volume, affine=np.eye(4))
        self.assertEqual(out.shape, volume.shape)

    def test_single_axis_flip(self):
        transform = RandomFlip(axes=2, flip_probability=1.0)
        volume = np.random.rand(2, 3, 4)
        flipped = transform(volume)
        expected = np.flip(volume, axis=2)
        np.testing.assert_array_equal(flipped, expected)

    def test_output_type_and_shape_consistency(self):
        transform = RandomFlip(axes=(0, 1), flip_probability=1.0)
        
        for dtype in [np.int16, np.int32, np.int64, np.float16, 
                      np.float32, np.float64, np.uint8, np.uint16]:
            flipped_np = transform(self.volume_np.astype(dtype))
            self.assertEqual(flipped_np.dtype, dtype)
            self.assertEqual(flipped_np.shape, self.volume_np.shape)


        for dtype in [torch.int16, torch.int32, torch.int64, 
                    torch.float16, torch.float32, torch.float64]:
            flipped_torch = transform(self.volume_torch.to(dtype))
            self.assertEqual(flipped_torch.dtype, dtype)
            self.assertEqual(flipped_torch.shape, self.volume_torch.shape)


class TestRandomErasing(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 20, 20, 20)
        self.numpy_input = np.random.rand(*self.shape)
        self.tensor_input = torch.tensor(self.numpy_input.copy())

    def test_output_type_and_shape_numpy(self):
        value = np.pi
        for dtype in [np.int16, np.int32, np.int64, np.float16, 
                      np.float32, np.float64, np.uint8, np.uint16]:
            transform = RandomErasing(value=value)
            out = transform(self.numpy_input.astype(dtype))
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, self.shape)

    def test_output_type_and_shape_tensor(self):
        value = torch.pi
        for dtype in [torch.int16, torch.int32, torch.int64, 
                    torch.float16, torch.float32, torch.float64]:
            transform = RandomErasing(value=value)
            out = transform(self.tensor_input.to(dtype))
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, self.shape)

    def test_inplace_true(self):
        data = np.ones(self.shape)
        transform = RandomErasing(inplace=True, value=5.0)
        out = transform(data)
        self.assertIs(out, data)
        self.assertFalse(np.allclose(out, 1.0))  # Some modification

    def test_inplace_false(self):
        data = np.ones(self.shape)
        transform = RandomErasing(inplace=False, value=5.0)
        out = transform(data)
        self.assertIsNot(out, data)
        self.assertEqual(out.shape, data.shape)

    def test_scalar_erasing(self):
        value, scale = 7.0, 0.3
        for _ in range(10):
            transform = RandomErasing(scale=(scale, scale), value=value)
            out = transform(np.ones(self.shape))
            nb_erased_values = (out == value).sum()
            nb_erased_values_th = round(scale * np.prod(self.shape))
            # Authorize +/- 5% of margin due to roundings 
            self.assertTrue(np.abs(nb_erased_values_th - nb_erased_values) < 0.05 * np.prod(self.shape))

    def test_mean_erasing(self):
        np.random.seed(42)
        data = np.random.rand(*self.shape)
        data_tensor = torch.rand(*self.shape)
        transform = RandomErasing(value="mean")
        out = transform(data)
        out_tensor = transform(data_tensor)
        diff = np.abs(out - data)
        diff_tensor = torch.abs(out_tensor - data_tensor)
        self.assertTrue(diff.sum() > 0)
        self.assertTrue(diff_tensor.sum() > 0)
        self.assertTrue(np.abs(out.mean() - data.mean()) < 1e-7)
        self.assertTrue(torch.abs(out_tensor.mean() - data_tensor.mean()) < 1e-7)

    def test_random_erasing(self):
        np.random.seed(42)
        data = np.zeros(self.shape)
        scale = (0.1, 0.5)
        transform = RandomErasing(scale=scale, value="random")
        for data in [np.zeros(self.shape), torch.zeros(self.shape)]:
            out = transform(data)
            nonzero_count = (out != 0).sum()
            nb_min_erased_values = round(scale[0] * np.prod(self.shape))
            nb_max_erased_values = round(scale[1] * np.prod(self.shape))
            self.assertTrue(nonzero_count >= nb_min_erased_values)
            self.assertTrue(nonzero_count <= nb_max_erased_values)

    def test_multiple_iterations(self):
        data = np.zeros(self.shape)
        value, scale = 9.0, (0.05, 0.2)
        transform = RandomErasing(scale=scale, value=value, num_iterations=3)
        nb_min_erased_values = round(scale[0] * np.prod(self.shape))
        nb_max_erased_values = 3 * round(scale[1] * np.prod(self.shape))
        out = transform(data)
        self.assertGreater((out == value).sum(), nb_min_erased_values)
        self.assertLess((out == value).sum(), nb_max_erased_values)

    def test_invalid_value(self):
        transform = RandomErasing(value="bad_value")
        with self.assertRaises(ValueError):
            transform(np.zeros(self.shape))
        with self.assertRaises(TypeError):
            transform = RandomErasing(ratio=2)
        with self.assertRaises(TypeError):
            transform = RandomErasing(scale=-1)
        with self.assertRaises(ValueError):
            transform = RandomErasing(scale=(-1, 2))
    
    def test_small_volume(self):
        transform = RandomErasing()
        for data in [np.ones((1, 2, 2, 2)), np.ones((1, 1, 1, 1)),  np.ones((2, 1, 1))]:
            out = transform(data)
            self.assertEqual(out.shape, data.shape)
            self.assertEqual(out.dtype, data.dtype)
    
    def test_large_ratio(self):
        transform = RandomErasing(ratio=(20, 30))
        out = transform(np.ones((2, 3, 4), dtype=np.float32))
        self.assertTrue((out != 1).sum() > 0)
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertEqual(out.dtype, np.float32)

    @patch("nidl.volume.transforms.augmentation.spatial.RandomErasing._sample_3d_box", 
           return_value=[slice(0, 3), slice(0, 3), slice(0, 3)])
    def test_mocked_box_numpy(self, mock_sample):
        data = np.ones((1, 10, 10, 10))
        transform = RandomErasing(value=2.0)
        out = transform(data)
        self.assertTrue(np.all(out[0, 0:3, 0:3, 0:3] == 2.0))

    @patch("nidl.volume.transforms.augmentation.spatial.RandomErasing._sample_3d_box", 
           return_value=[slice(0, 3), slice(0, 3), slice(0, 3)])
    def test_mocked_box_tensor(self, mock_sample):
        data = torch.ones((1, 10, 10, 10))
        transform = RandomErasing(value=3.0)
        out = transform(data)
        self.assertTrue(torch.all(out[0, 0:3, 0:3, 0:3] == 3.0))


class TestRandomResizedCrop(unittest.TestCase):

    def setUp(self):
        self.shape = (32, 32, 32)
        self.channel_shape = (1, 32, 32, 32)
        self.target_shape = (16, 14, 12)
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def test_output_type_and_shape_numpy(self):
        arr = np.random.rand(*self.shape)
        for dtype in [np.int16, np.int32, np.int64, # float16 raises Error
                      np.float32, np.float64, np.uint8, np.uint16]:
            transform = RandomResizedCrop(target_shape=self.target_shape)
            out = transform(arr.astype(dtype))
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, self.target_shape)
            self.assertEqual(out.dtype, dtype)

    def test_output_type_and_shape_torch(self):
        tensor = torch.rand(*self.shape)
        for dtype in [torch.int16, torch.int32, torch.int64, # float16 raises Error
                      torch.float32, torch.float64]:
            transform = RandomResizedCrop(target_shape=self.target_shape)
            out = transform(tensor.to(dtype))
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.shape, torch.Size(self.target_shape))
            self.assertEqual(out.dtype, dtype)

    def test_channel_preservation_numpy(self):
        arr = np.random.rand(*self.channel_shape)
        transform = RandomResizedCrop(target_shape=self.target_shape)
        out = transform(arr)
        self.assertEqual(out.shape, (1,) + self.target_shape)

    def test_channel_preservation_torch(self):
        tensor = torch.rand(*self.channel_shape)
        transform = RandomResizedCrop(target_shape=self.target_shape)
        out = transform(tensor)
        self.assertEqual(out.shape, torch.Size((1,) + self.target_shape))

    def test_interpolation(self):
        arr = np.random.rand(*self.shape).astype(np.float32)
        for interp in ['nearest', 'linear', 'bspline', 'cubic', \
                    'gaussian', 'label_gaussian', 'hamming', 'cosine', \
                    'welch', 'lanczos', 'blackman']:
            transform = RandomResizedCrop(target_shape=self.target_shape, 
                                          interpolation=interp)
            out = transform(arr)
            self.assertEqual(out.shape, self.target_shape)
            self.assertEqual(out.dtype, np.float32)

    def test_invalid_target_shape(self):
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(-1, 20, 20))
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(1, 20, 20, 20))
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(20.0, 20, 20))

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(16, 16, 16), scale=(1.2, 0.5))
        with self.assertRaises(TypeError):
            RandomResizedCrop(target_shape=(16, 16, 16), scale=(1.2, "3.2"))
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(16, 16, 16), scale=(0.5, 2.0))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=17.2)
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=17.0)
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(16, 17))

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            RandomResizedCrop(target_shape=(16, 16, 16), ratio=(2.0, 0.5))

    def test_output_value_range(self):
        arr = np.random.rand(*self.shape)
        transform = RandomResizedCrop(target_shape=self.target_shape)
        out = transform(arr)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
    
    def test_sample_3d_box_returns_full_volume(self):
        in_shape = (10, 10, 10)
        # Use an impossible scale so cbrt(target_volume) > each dim even with ratio=1
        slices = RandomResizedCrop._sample_3d_box(
            in_shape=in_shape, scale=(2.0, 2.0), ratio=(1.0, 1.0)
        )
        assert slices == [slice(0, 10), slice(0, 10), slice(0, 10)]

    def test_multiple_runs_different_results(self):
        arr = np.random.rand(*self.shape)
        transform = RandomResizedCrop(target_shape=17)
        out1 = transform(arr)
        out2 = transform(arr)
        # Not guaranteed to differ, but very likely
        self.assertFalse(np.allclose(out1, out2), msg="Transforms should be stochastic.")


class TestRandomGaussianBlur(unittest.TestCase):
    def setUp(self):
        self.shape = (32, 16, 64)
        self.channel_shape = (1, 32, 16, 64)
    
    def test_output_type_and_shape_numpy(self):
        transform = RandomGaussianBlur(sigma=(0.1, 2))
        for shape in [self.shape, self.channel_shape]:
            for dtype in [np.int16, np.int32, np.int64,  # float16 raises Error
                          np.float32, np.float64, np.uint8, np.uint16]:
                arr = np.random.rand(*shape)
                out = transform(arr.astype(dtype))
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.shape, shape)
                self.assertEqual(out.dtype, dtype)

    def test_output_type_and_shape_tensor(self):
        transform = RandomGaussianBlur(sigma=(1.5, 3.8))
        for shape in [self.shape, self.channel_shape]:
            for dtype in [torch.int16, torch.int32, torch.int64, 
                          torch.float32, torch.float64]: # float16 raises Error
                arr = torch.rand(*shape)
                out = transform(arr.to(dtype))
                self.assertIsInstance(out, torch.Tensor)
                self.assertEqual(out.shape, shape)
                self.assertEqual(out.dtype, dtype)

    def test_sigma_range_valid_2d(self):
        tf = RandomGaussianBlur(sigma=[0.1, 1.0])
        self.assertEqual(tf.sigma, (0.1, 1.0, 0.1, 1.0, 0.1, 1.0))

    def test_sigma_range_valid_6d(self):
        tf = RandomGaussianBlur(sigma=(0.1, 1.0, 0.2, 1.2, 0.3, 1.3))
        self.assertEqual(tf.sigma, (0.1, 1.0, 0.2, 1.2, 0.3, 1.3))

    def test_invalid_sigma_length(self):
        with self.assertRaises(ValueError):
            RandomGaussianBlur(sigma=(0.1, 1.0, 0.5))
        with self.assertRaises(ValueError):
            RandomGaussianBlur(sigma=(0.1, 1.0, 0.5, 1.0))

    def test_invalid_sigma_values(self):
        with self.assertRaises(ValueError):
            RandomGaussianBlur(sigma=(-1, 0.5))  # negative sigma

    def test_blur_does_change_data(self):
        vol = np.ones((16, 16, 16))
        vol[8, 8, 8] = 10  # sharp spike
        tf = RandomGaussianBlur(sigma=(3.0, 3.0))
        blurred = tf(vol)
        # Local change only
        self.assertLess(blurred[8, 8, 8], 10)
        self.assertGreater(blurred[8, 9, 10], 1)
        self.assertTrue(np.abs(blurred[0,0,0] - 1) < 1e-3)


class TestRandomGaussianNoise(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 10, 10, 10)
        self.numpy_input = np.ones(self.shape, dtype=np.float32)
        self.torch_input = torch.ones(self.shape, dtype=torch.float32)

    def test_output_type_and_shape_tensor(self):
        transform = RandomGaussianNoise(mean=0, std=(0.1, 3.0))
        for shape in [self.shape, self.shape[1:]]:
            for dtype in [torch.int16, torch.int32, torch.int64, 
                          torch.float16, torch.float32, torch.float64]:
                output = transform(torch.rand(*shape).to(dtype))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, shape)

    def test_output_type_and_shape_numpy(self):
        transform = RandomGaussianNoise(mean=0, std=(0.1, 3.0))
        for shape in [self.shape, self.shape[1:]]:
            for dtype in [np.int16, np.int32, np.int64, np.float16, 
                          np.float32, np.float64, np.uint8, np.uint16]:
                output = transform(np.random.rand(*shape).astype(dtype))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, shape)

    def test_statistical_effect_numpy(self):
        transform = RandomGaussianNoise(mean=0, std=(0.1, 0.1))
        output = transform(self.numpy_input)
        self.assertAlmostEqual(np.std(output), 0.1, places=2)

    def test_statistical_effect_torch(self):
        transform = RandomGaussianNoise(mean=0, std=(0.1, 0.1))
        output = transform(self.torch_input)
        self.assertAlmostEqual(torch.std(output).item(), 0.1, places=2)

    def test_different_mean_range(self):
        transform = RandomGaussianNoise(mean=(-1.0, 1.0), std=(0.1, 0.1))
        output = transform(self.numpy_input)
        self.assertGreater(np.mean(output), -1 + np.mean(self.numpy_input))
        self.assertLess(np.mean(output), 1 + np.mean(self.numpy_input))
        output = transform(self.torch_input)
        self.assertGreater(torch.mean(output), -1 + torch.mean(self.torch_input))
        self.assertLess(torch.mean(output), 1 + torch.mean(self.torch_input))

    def test_different_std_range(self):
        transform = RandomGaussianNoise(mean=(-1.0, 1.0), std=(3.0, 10.0))
        np_input = np.random.rand(3, 4, 5)
        torch_input = torch.rand(3, 4, 5)
        output = transform(self.numpy_input)
        self.assertGreater(np.std(output), np.std(np_input))
        self.assertLess(np.std(output), np.sqrt(10**2 + np.var(np_input)))
        output = transform(torch_input)
        self.assertGreater(torch.std(output), torch.std(torch_input))
        self.assertLess(torch.std(output), torch.sqrt(10**2 + torch.var(torch_input)))

    def test_invalid_std_range_raises(self):
        with self.assertRaises(ValueError):
            RandomGaussianNoise(mean=0, std=(-1.0, 0.1))


class TestZNormalization(unittest.TestCase):
    def setUp(self):
        self.shape_3d = (10, 10, 10)
        self.shape_4d = (3, 10, 10, 10)
        self.eps = 1e-8

        self.numpy_input_3d = np.random.rand(*self.shape_3d)
        self.numpy_input_4d = np.random.rand(*self.shape_4d)
        self.torch_input_3d = torch.rand(self.shape_3d)
        self.torch_input_4d = torch.rand(self.shape_4d)

    def test_mean_std_numpy_3d(self):
        transform = ZNormalization()
        out = transform(self.numpy_input_3d)
        self.assertAlmostEqual(np.mean(out), 0, places=3)
        self.assertAlmostEqual(np.std(out), 1, places=3)

    def test_mean_std_numpy_4d(self):
        transform = ZNormalization()
        out = transform(self.numpy_input_4d)
        for c in range(out.shape[0]):
            self.assertAlmostEqual(np.mean(out[c]), 0, places=3)
            self.assertAlmostEqual(np.std(out[c]), 1, places=3)

    def test_mean_std_torch_4d(self):
        transform = ZNormalization()
        out = transform(self.torch_input_4d)
        for c in range(out.shape[0]):
            self.assertAlmostEqual(torch.mean(out[c]).item(), 0, places=3)
            self.assertAlmostEqual(torch.std(out[c]).item(), 1, places=3)

    def test_masking_function_numpy(self):
        def mask_fn(x): return x > 0.5
        transform = ZNormalization(masking_fn=mask_fn)
        mask = mask_fn(self.numpy_input_3d)
        out = transform(self.numpy_input_3d)
        self.assertEqual(out.shape, self.numpy_input_3d.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(out[mask].mean(), 0, places=3)
        self.assertAlmostEqual(out[mask].std(), 1, places=3)

    def test_masking_function_torch(self):
        def mask_fn(x): return x > 0.5
        transform = ZNormalization(masking_fn=mask_fn)
        mask = mask_fn(self.torch_input_3d)
        out = transform(self.torch_input_3d)
        self.assertEqual(out.shape, self.torch_input_3d.shape)
        self.assertTrue(torch.all(torch.isfinite(out)))
        self.assertAlmostEqual(out[mask].mean().item(), 0, places=3)
        self.assertAlmostEqual(out[mask].std().item(), 1, places=3)

    def test_small_variance(self):
        # Input with nearly constant values
        input_data = np.full(self.shape_3d, fill_value=3.14, dtype=np.float32)
        transform = ZNormalization()
        out = transform(input_data)
        self.assertTrue(np.std(out) < 1e-6) # Very small value
        self.assertTrue(np.all(np.isfinite(out)))

    def test_preserves_dtype_and_shape_numpy(self):
        transform = ZNormalization()
        for shape in [self.shape_3d, self.shape_4d]:
            for dtype in [np.float16, np.float32, np.float64]: # int raises Error
                output = transform(np.random.rand(*shape).astype(dtype))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, shape)
    
    def test_preserves_dtype_and_shape_tensor(self):
        transform = ZNormalization()
        for shape in [self.shape_3d, self.shape_4d]:
            for dtype in [torch.float16, torch.float32, torch.float64]: # int raises Error
                output = transform(torch.rand(*shape).to(dtype))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, shape)


class TestRobustRescaling(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def test_numpy_single_channel_default(self):
        volume = np.random.normal(100, 20, (64, 64, 64)).astype(np.float32)
        transform = RobustRescaling()
        output = transform(volume)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, volume.shape)
        self.assertTrue(np.all(output >= 0.0) and np.all(output <= 1.0))
        self.assertEqual(output.dtype, volume.dtype)

    def test_numpy_multi_channel_custom_range(self):
        volume = np.random.normal(10, 5, (2, 32, 32, 32)).astype(np.float32)
        transform = RobustRescaling(out_min_max=(-1, 1), percentiles=(5, 95))
        output = transform(volume)
        self.assertEqual(output.shape, volume.shape)
        self.assertTrue(np.all(output >= -1.0) and np.all(output <= 1.0))

    def test_torch_single_channel(self):
        volume = torch.randn(64, 64, 64, dtype=torch.float32) * 5 + 100
        transform = RobustRescaling(out_min_max=(0, 1), percentiles=(1, 99))
        output = transform(volume)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, volume.shape)
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))
        self.assertEqual(output.dtype, volume.dtype)

    def test_torch_multi_channel(self):
        volume = torch.randn(3, 32, 32, 32, dtype=torch.float32) * 10 + 50
        transform = RobustRescaling(out_min_max=(0, 1), percentiles=(1, 99))
        output = transform(volume)
        self.assertEqual(output.shape, volume.shape)
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_outliers_are_clipped(self):
        volume = np.random.normal(50, 5, (64, 64, 64)).astype(np.float32)
        volume[0, 0, 0] = 1e6  # very large outlier
        transform = RobustRescaling(out_min_max=(0, 1), percentiles=(1, 99))
        output = transform(volume)
        self.assertTrue(np.all(output >= 0.0) and np.all(output <= 1.0))

    def test_masking_function_numpy(self):
        volume = np.random.uniform(0, 100, (64, 64, 64)).astype(np.float32)
        mask = np.zeros_like(volume, dtype=bool)
        mask[16:48, 16:48, 16:48] = True

        def masking_fn(data):
            return mask

        transform = RobustRescaling(masking_fn=masking_fn)
        output = transform(volume)
        self.assertEqual(output.shape, volume.shape)
        self.assertTrue(np.all(output >= 0.0) and np.all(output <= 1.0))

    def test_masking_function_torch(self):
        volume = torch.rand(1, 32, 32, 32, dtype=torch.float32) * 100

        def masking_fn(data):
            return data > 10

        transform = RobustRescaling(masking_fn=masking_fn)
        output = transform(volume)
        self.assertEqual(output.shape, volume.shape)
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_output_type_and_shape_tensor(self):
        transform = RobustRescaling()
        for shape in [(1, 10, 20, 30), (10, 20, 20)]:
            for dtype in [torch.int16, torch.int32, torch.int64, torch.float16, 
                          torch.float32, torch.float64]:
                output = transform(torch.rand(*shape).to(dtype))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, shape)
                self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_output_type_and_shape_numpy(self):
        transform = RobustRescaling()
        for shape in [(1, 10, 20, 30), (10, 20, 20)]:
            for dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, 
                        np.float64, np.uint8, np.uint16]:
                output = transform(np.random.rand(*shape).astype(dtype))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, shape)
                self.assertTrue(np.all(output >= 0.0) and np.all(output <= 1.0))

    def test_invalid_percentiles(self):
        with self.assertRaises(ValueError):
            RobustRescaling(percentiles=(90,))  # only one value

        with self.assertRaises(ValueError):
            RobustRescaling(percentiles=(99, 1))  # min > max

        with self.assertRaises(ValueError):
            RobustRescaling(percentiles=(-5, 105))  # out of range

    def test_output_range_edge_case(self):
        volume = np.full((64, 64, 64), 5.0, dtype=np.float32)
        transform = RobustRescaling(out_min_max=(0, 1), percentiles=(0, 100))
        output = transform(volume)
        # All values should be exactly 0 because min == max
        self.assertTrue(np.allclose(output, 0.0, atol=1e-6))


class TestCropOrPad(unittest.TestCase):
    def setUp(self):
        self.shape_3d = (16, 29, 32)

    def _generate_data(self, shape, kind='np', fill_value=1):
        if kind == 'np':
            return np.full(shape, fill_value, dtype=np.float32)
        elif kind == 'torch':
            return torch.full(shape, fill_value, dtype=torch.float32)
        raise ValueError("Unknown kind")

    def test_no_op_same_shape(self):
        for kind in ['np', 'torch']:
            input_data = self._generate_data(self.shape_3d, kind)
            transform = CropOrPad(target_shape=self.shape_3d)
            output = transform(input_data)
            self.assertEqual(output.shape, input_data.shape)
            self.assertTrue(np.allclose(output, input_data))

    def test_crop(self):
        shape = (40, 40, 40)
        target = (32, 32, 32)
        for kind in ['np', 'torch']:
            input_data = self._generate_data(shape, kind, fill_value=5)
            transform = CropOrPad(target_shape=target)
            output = transform(input_data)
            self.assertEqual(output.shape, target)
            self.assertTrue(np.allclose(output, 5))

    def test_pad(self):
        shape = (24, 24, 24)
        target = (32, 32, 32)
        for kind in ['np', 'torch']:
            input_data = self._generate_data(shape, kind, fill_value=3)
            transform = CropOrPad(target_shape=target, constant_values=0)
            output = transform(input_data)
            self.assertEqual(output.shape, target)
            center = output[12:-24, 24:-24, 24:-24]
            self.assertTrue(np.allclose(center, 3))
            self.assertTrue((output == 0).any())

    def test_crop_and_pad(self):
        shape = (40, 24, 24)
        target = (32, 32, 32)
        for kind in ['np', 'torch']:
            input_data = self._generate_data(shape, kind, fill_value=7)
            transform = CropOrPad(target_shape=target, constant_values=-1)
            output = transform(input_data)
            self.assertEqual(output.shape, target)
            self.assertTrue((output == 7).any())
            self.assertTrue((output == -1).any())

    def test_channel_dim_handling(self):
        shape = (1, 24, 24, 24)
        target = (32, 32, 32)
        for kind in ['np', 'torch']:
            input_data = self._generate_data(shape, kind, fill_value=2)
            transform = CropOrPad(target_shape=target, constant_values=0)
            output = transform(input_data)
            self.assertEqual(output.shape, (1, 32, 32, 32))
            self.assertTrue((output != 0).any())

    def test_padding_modes(self):
        shape = (8, 8, 8)
        target = (12, 12, 12)
        input_data = self._generate_data(shape, 'np')
        for dtype in [np.int16, np.int32, np.int64, np.float32, 
                                np.float64, np.uint8, np.uint16]:
            for padding_mode in ['edge', 'maximum', 'constant', 'mean', 'median',
                                'minimum', 'reflect', 'symmetric']:
                transform = CropOrPad(target_shape=target, padding_mode=padding_mode)
                output = transform(input_data.astype(dtype))
                self.assertEqual(output.shape, target)
                self.assertEqual(output.dtype, dtype)

    def test_invalid_shape_length(self):
        input_data = self._generate_data((32, 32, 32))
        with self.assertRaises(ValueError):
            transform = CropOrPad(target_shape=(32, 32))  # invalid

    def test_preserves_dtype_and_shape_numpy(self):
        for dtype in [np.int16, np.int32, np.int64, np.float16, 
                      np.float32, np.float64, np.uint8, np.uint16]:
            transform = CropOrPad(target_shape=(3, 32, 32))
            out = transform(np.random.random((1, 3, 4, 5)).astype(dtype))
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_preserves_dtype_and_shape_tensor(self):
        for dtype in [torch.int16, torch.int32, torch.int64, torch.float16, 
                      torch.float32, torch.float64, torch.uint8, torch.uint16]:
            transform = CropOrPad(target_shape=(3, 32, 32))
            out = transform(torch.rand(1, 3, 4, 5).to(dtype))
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, (1, 3, 32, 32))


class TestResample(unittest.TestCase):
    def setUp(self):
        self.image_3d = np.random.rand(32, 32, 32).astype(np.float32)
        self.image_4d = np.random.rand(1, 32, 32, 32).astype(np.float32)
        self.affine = np.eye(4)

    def test_resample_numpy_3d(self):
        resampler = Resample(target=2.0)
        out = resampler(self.image_3d, self.affine)
        self.assertEqual(out.shape, (16, 16, 16))
        self.assertIsInstance(out, np.ndarray)

    def test_resample_numpy_4d(self):
        resampler = Resample(target=(1.0, 3.0, 3.0))
        out = resampler(self.image_4d, self.affine)
        self.assertEqual(out.shape[0], 1)
        self.assertIsInstance(out, np.ndarray)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 1))

    def test_resample_torch(self):
        img = torch.from_numpy(self.image_3d)
        resampler = Resample(target=2.0)
        out = resampler(img, self.affine)
        self.assertEqual(out.shape, (16, 16, 16))
        self.assertIsInstance(out, torch.Tensor)
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_invalid_affine_shape(self):
        with self.assertRaises(ValueError):
            Resample()(self.image_3d, affine=np.eye(3))

    def test_invalid_affine_orientation(self):
        affine = np.diag([1, 1, -1, 1])  # Not RAS
        with self.assertRaises(ValueError):
            Resample()(self.image_3d, affine=affine)

    def test_invalid_interpolation(self):
        with self.assertRaises(ValueError):
            Resample(target=1.0, interpolation="invalid")

    def test_invalid_spacing_type(self):
        with self.assertRaises(ValueError):
            Resample(target="not_a_number")

    def test_invalid_spacing_value(self):
        with self.assertRaises(ValueError):
            Resample(target=-1.0)

    def test_preserves_dtype_and_shape_numpy(self):
        for dtype in [np.int16, np.int32, np.int64, 
                      np.float32, np.float64, np.uint8, np.uint16]:
            transform = Resample(target=(2, 3, 2))
            out = transform(np.random.random((1, 16, 9, 8)).astype(dtype))
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, (1, 8, 3, 4))

    def test_preserves_dtype_and_shape_tensor(self):
        for dtype in [torch.int16, torch.int32, torch.int64, 
                      torch.float32, torch.float64, torch.uint8, 
                      torch.uint16]:
            transform = Resample(target=(2, 3, 2))
            out = transform(torch.rand(1, 16, 9, 8).to(dtype))
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, (1, 8, 3, 4))

if __name__ == "__main__":
    unittest.main()
