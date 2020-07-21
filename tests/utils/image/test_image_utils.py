import unittest
import os
import numpy as np
import tensorflow as tf
from simple_sr.utils.image import image_utils, metrics

DUMP_RECONSTRUCTED = True
DATA_DIR = "./tests/data"
SAVE_DIR = "./tests/data/reconstructed"
PATCH_DIMS = [(1, 1), (2, 2), (3, 3), (3, 1), (1, 3), (2, 3), (3, 2)]


class TestImageUtils(unittest.TestCase):

    def setUp(self):
        self.mat_3x3 = tf.constant([
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
            [[7, 7, 7], [8, 8, 8], [9, 9, 9]]
        ])

        self.mat_5x3 = tf.constant([
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]],
            [[11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15]]
        ])
        self.matrices = [
            self.mat_3x3,
            self.mat_5x3
        ]

        if DUMP_RECONSTRUCTED:
            os.makedirs(SAVE_DIR, exist_ok=True)
        ds = tf.data.Dataset.list_files(
            [os.path.join(DATA_DIR, fname) for fname in os.listdir(DATA_DIR)
             if os.path.isfile(os.path.join(DATA_DIR, fname))]
        )
        ds = ds.map(tf.io.read_file)
        ds = ds.map(tf.image.decode_png).batch(1)
        self.images = ds

    def test_segmentation(self):
        for patch_dim in PATCH_DIMS:
            for matrix in self.matrices:
                patches, padding = image_utils.segment_into_patches(
                    matrix, patch_width=patch_dim[0], patch_height=patch_dim[1]
                )
                self.assertEqual(4, patches.shape.rank)
                self.assertEqual(patch_dim[0], patches.shape[2])
                self.assertEqual(patch_dim[1], patches.shape[1])

    def test_segment_and_reconstruct(self):
        for patch_dim in PATCH_DIMS:
            for matrix in self.matrices:
                patches, padding = image_utils.segment_into_patches(
                    matrix, patch_width=patch_dim[0], patch_height=patch_dim[1]
                )

                reconstructed = image_utils.reconstruct_from_patches(
                    patches, original_height=matrix.shape[0], original_width=matrix.shape[1],
                    horizontal_padding=padding[0][1], vertical_padding=padding[1][1]
                )
                self._assert_shape(matrix, reconstructed)

    def test_segment_with_overlap_and_reconstruct(self):
        patch_dims = ((32, 32), (64, 64), (128, 128))
        for idx, img in enumerate(self.images):
            for patch_dim in patch_dims:
                pixel_overlap = patch_dim[0] // 4
                patches, padding = image_utils.segment_into_patches(
                    img, patch_width=patch_dim[0], patch_height=patch_dim[1],
                    pixel_overlap=pixel_overlap
                )

                reconstructed = image_utils.reconstruct_from_overlapping_patches(
                    patches, image_height=img.shape[1], image_width=img.shape[2],
                    pixel_overlap=pixel_overlap,
                    horizontal_padding=(padding[0][1] - pixel_overlap),
                    vertical_padding=(padding[1][1] - pixel_overlap)
                )
                _img = img
                if _img.shape.rank == 4:
                    _img = tf.reshape(_img, (_img.shape[1:]))
                self._assert_shape(_img, reconstructed)
                self._assert_content(_img, reconstructed)
                self._dump_reconstructed(reconstructed, patch_dim, idx)

    def test_segment_and_reconstruct_real_image(self):
        patch_dims = ((32, 32), (64, 64), (128, 128))
        for idx, img in enumerate(self.images):
            img = image_utils._extract_tensor(img)
            for patch_dim in patch_dims:
                patches, padding = image_utils.segment_into_patches(
                    img, patch_width=patch_dim[0], patch_height=patch_dim[1]
                )

                reconstructed = image_utils.reconstruct_from_patches(
                    patches, original_height=img.shape[0], original_width=img.shape[1],
                    horizontal_padding=padding[0][1], vertical_padding=padding[1][1]
                )
                _img = img
                if _img.shape.rank == 4:
                    _img = tf.reshape(_img, (_img.shape[1:]))
                self._assert_shape(_img, reconstructed)
                self._assert_content(_img, reconstructed)
                self._dump_reconstructed(reconstructed, patch_dim, idx)

    def _dump_reconstructed(self, reconstructed, patch_dims, idx):
        if DUMP_RECONSTRUCTED:
            recon_img = image_utils.tensor_to_img(reconstructed)
            recon_img.save(f"{SAVE_DIR}/recon{idx}_{patch_dims[0]}x{patch_dims[1]}.png")

    def _assert_content(self, original, reconstructed):
        self.assertEqual(0, tf.keras.metrics.MeanSquaredError()(original, reconstructed))
        self.assertEqual(float("inf"), metrics.psnr(original, reconstructed, max_val=255))
        self.assertEqual(1.0, metrics.ssim(original, reconstructed, max_val=255))
        self.assertEqual(float("inf"), metrics.psnr_on_y((original/255), (reconstructed/255)))

    def _assert_shape(self, original, reconstructed):
        self.assertEqual(3, reconstructed.shape.rank)
        self.assertEqual(original.shape[-3], reconstructed.shape[-3])
        self.assertEqual(original.shape[-2], reconstructed.shape[-2])
        self.assertEqual(original.shape[-1], reconstructed.shape[-1])
        _original = original
        if original.shape.rank == 4:
            _original = tf.reshape(_original, (_original.shape[1:]))
        np.testing.assert_array_equal(_original.numpy(), reconstructed.numpy())


if __name__ == "__main__":
    unittest.main()
