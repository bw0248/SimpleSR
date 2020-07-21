import unittest
import tensorflow as tf

from simple_sr.data_pipeline.data_pipeline import DataPipeline
from simple_sr.utils.image import metrics, image_transforms

LR_DIMS = (32, 32)
HR_DIMS = (64, 64)
SCALE = 2
NUM_IMGS = 8
DATA_PATH = [
    "./tests/data/patterns/random_noise",
    "./tests/data/patterns/gradients"
]
AUGMENTATIONS = [image_transforms.rotate90, image_transforms.flip_along_y]
NUM_CROPS = 8
TRAIN_VAL_SPLIT = 0.25
BATCH_SIZES = [8, 16]


class TestDataPipeline(unittest.TestCase):

    def test_lr_hr_correspondence(self):
        for batch_size in BATCH_SIZES:
            for pipeline in self._load_pipelines(batch_size):
                for generator in self._get_generators(pipeline):
                    self._assert_lr_hr_correspondence(generator)

    def test_num_imgs(self):
        for batch_size in BATCH_SIZES:
            for pipeline in self._load_pipelines(batch_size):
                self._assert_num_imgs(
                    pipeline.train_batch_generator(),
                    NUM_IMGS * len(DATA_PATH) * (1 - TRAIN_VAL_SPLIT) * NUM_CROPS * (len(AUGMENTATIONS) + 1)
                )
                self._assert_num_imgs(
                    pipeline.validation_batch_generator(),
                    NUM_IMGS * len(DATA_PATH) * TRAIN_VAL_SPLIT * NUM_CROPS
                )

    def test_batch_shapes(self):
        for batch_size in BATCH_SIZES:
            for pipeline in self._load_pipelines(batch_size):
                self._assert_batch_shapes(
                    pipeline.train_batch_generator(), batch_size,
                    int((NUM_IMGS * len(DATA_PATH) * (1 - TRAIN_VAL_SPLIT) * NUM_CROPS * (len(AUGMENTATIONS) + 1))
                        / batch_size)
                )
                self._assert_batch_shapes(
                    pipeline.validation_batch_generator(), batch_size,
                    int((NUM_IMGS * len(DATA_PATH) * TRAIN_VAL_SPLIT * NUM_CROPS) / batch_size)
                )

    def test_train_val_split(self):
        pipeline_with_split = DataPipeline(
            hr_img_path=DATA_PATH, scale=2,
            train_val_split=TRAIN_VAL_SPLIT, batch_size=8,
            crop_naive=True, crop_size=(*HR_DIMS, 3), num_crops=NUM_CROPS,
            augmentations=AUGMENTATIONS
        )
        self._assert_train_val_split(pipeline_with_split)

        pipeline_with_separate_val_path = DataPipeline(
            hr_img_path=DATA_PATH[0], scale=2,
            validationset_path=DATA_PATH[1],
            train_val_split=0, batch_size=8,
            crop_naive=True, crop_size=(*HR_DIMS, 3), num_crops=NUM_CROPS,
            augmentations=AUGMENTATIONS
        )
        self._assert_train_val_split(pipeline_with_separate_val_path)

    def test_valid_set_only(self):
        valid_pipeline = DataPipeline(
            hr_img_path=None, scale=2,
            validationset_path=DATA_PATH, train_val_split=0.0,
            crop_size=(*HR_DIMS, 3), num_crops=NUM_CROPS
        )
        self.assertEqual(0, len(valid_pipeline.train_imgs))
        self.assertNotEqual(0, len(valid_pipeline.val_imgs))
        self._assert_lr_hr_correspondence(valid_pipeline.validation_batch_generator())
        self._assert_num_imgs(
            valid_pipeline.validation_batch_generator(),
            NUM_IMGS * len(DATA_PATH) * NUM_CROPS
        )
        self._assert_num_imgs(valid_pipeline.train_batch_generator(), 0)
        self._assert_num_imgs(valid_pipeline.test_batch_generator(), 0)

    def test_test_set_only(self):
        test_pipeline = DataPipeline(
            hr_img_path=None, scale=None,
            validationset_path=None, train_val_split=0.0,
            test_img_paths=DATA_PATH
        )
        self.assertEqual(0, len(test_pipeline.train_imgs))
        self.assertEqual(0, len(test_pipeline.val_imgs))
        self._assert_num_imgs(
            test_pipeline.test_batch_generator(batch_size=1),
            NUM_IMGS * len(DATA_PATH)
        )
        self._assert_num_imgs(test_pipeline.train_batch_generator(), 0)
        self._assert_num_imgs(test_pipeline.validation_batch_generator(), 0)

    def test_train_set_only(self):
        train_pipeline = DataPipeline(
            hr_img_path=DATA_PATH, scale=2,
            validationset_path=None, train_val_split=0.0, crop=True,
            crop_size=(*HR_DIMS, 3), num_crops=NUM_CROPS,
            test_img_paths=None
        )
        self.assertNotEqual(0, len(train_pipeline.train_imgs))
        self.assertEqual(0, len(train_pipeline.val_imgs))
        self._assert_num_imgs(
            train_pipeline.train_batch_generator(),
            NUM_IMGS * len(DATA_PATH) * NUM_CROPS
        )
        self._assert_num_imgs(train_pipeline.test_batch_generator(), 0)
        self._assert_num_imgs(train_pipeline.validation_batch_generator(), 0)

    def _assert_train_val_split(self, pipeline):
        train_fnames = set(pipeline.train_imgs)
        val_fnames = set(pipeline.val_imgs)
        common_fnames = train_fnames & val_fnames
        self.assertEqual(0, len(common_fnames))

    def _assert_batch_shapes(self, generator, batch_size, expected_full_batch):
        for idx, (lr_batch, hr_batch) in enumerate(generator):
            # last batch may contain less samples
            if idx < expected_full_batch:
                self.assertEqual(lr_batch.shape[0], batch_size)
                self.assertEqual(hr_batch.shape[0], batch_size)

                self.assertEqual((lr_batch.shape[1], lr_batch.shape[2]), LR_DIMS)
                self.assertEqual((hr_batch.shape[1], hr_batch.shape[2]), HR_DIMS)

                self.assertEqual(lr_batch.shape[3], 3)
                self.assertEqual(hr_batch.shape[3], 3)

    def _assert_num_imgs(self, generator, expected_total_imgs):
        num_lr_imgs = 0
        num_hr_imgs = 0
        for (lr_batch, hr_batch) in generator:
            num_lr_imgs += lr_batch.shape[0]
            num_hr_imgs += hr_batch.shape[0]
        self.assertEqual(num_lr_imgs, expected_total_imgs)
        self.assertEqual(num_hr_imgs, expected_total_imgs)

    def _assert_lr_hr_correspondence(self, batch_generator):
        for (lr_batch, hr_batch) in batch_generator:
            hr_batch = self._denormalize_batch(hr_batch)
            # resize hr_batch for comparision to lr_batch
            hr_batch_resized = image_transforms.resize(
                hr_batch, LR_DIMS, resize_filter=tf.image.ResizeMethod.BICUBIC
            )
            for (lr_img, resized_hr_img) in zip(lr_batch, hr_batch_resized):
                ssim = metrics.ssim(lr_img, resized_hr_img)
                self.assertAlmostEqual(1.0, ssim, delta=1e-6)
                mse = tf.keras.metrics.MeanSquaredError()(lr_img, resized_hr_img)
                self.assertAlmostEqual(0.0, mse, delta=1e-10)

    def _denormalize_batch(self, hr_batch):
        return ((hr_batch + 1) * 127.5) / 255

    def _load_pipelines(self, batch_size):
        pipeline_crop_naive = DataPipeline(
            hr_img_path=DATA_PATH, scale=2,
            train_val_split=TRAIN_VAL_SPLIT, batch_size=batch_size,
            crop_naive=True, crop_size=(*HR_DIMS, 3), num_crops=NUM_CROPS,
            augmentations=AUGMENTATIONS
        )
        pipeline_crop_divers = DataPipeline(
            hr_img_path=DATA_PATH, scale=2,
            train_val_split=TRAIN_VAL_SPLIT, batch_size=batch_size,
            crop_naive=False, crop_size=(*HR_DIMS, 3), num_crops=NUM_CROPS,
            minimum_variation_patch=0.8, minimum_variation_batch=0.05,
            augmentations=AUGMENTATIONS
        )
        return [pipeline_crop_naive, pipeline_crop_divers]

    def _get_generators(self, pipeline):
        return [
            pipeline.train_batch_generator(),
            pipeline.validation_batch_generator()
        ]


if __name__ == "__main__":
    unittest.main()
