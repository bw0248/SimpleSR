import unittest
import os
import tensorflow as tf

from simple_sr.utils.config.config_util import ConfigUtil
from simple_sr.utils.image import image_transforms
from simple_sr.utils.models.loss_functions.mean_squared_error import MeanSquaredError

TRAIN_DATA_PATHS = ["./data/patterns/gradients", "./data/patterns/random_noise"]
VAL_DATA_PATHS = ["./data/patterns/gradients"]
AUGMENTATIONS = ["rotate90", "flip_along_y", "adjust_contrast"]
TEST_DATA_PATH = ["./data/patterns/random_noise"]
TEST_ORIGINALS_PATH = None  # todo
RESULTS_SAVE_PATH = "./results"
RESIZE_FILTER = "gaussian"
OPTIMIZER = "Adam"

LOSS_FUNCTIONS = [
    {"loss_function": "MeanSquaredError", "weighted": False, "loss_weight": 1.0},
    {"loss_function": "MeanAbsoluteError", "weighted": True, "loss_weight": 2.0}
]

RESIZE_FILTER_TO_OBJ = {
    "gaussian": tf.image.ResizeMethod.GAUSSIAN,
    "bicubic": tf.image.ResizeMethod.BICUBIC,
    "bilinear": tf.image.ResizeMethod.BILINEAR
}

OPTIMIZER_TO_OBJ = {
    "Adam": tf.keras.optimizers.Adam
}

AUGMENTATION_TO_OBJ = {func.__name__: func
                       for func in image_transforms.get_all_available_augmentations()}


class TestConfigUtil(unittest.TestCase):
    def test_train_config_from_yaml(self):
        config_yaml = self._prepare_train_yaml()
        config, pipeline, model = ConfigUtil.from_yaml(config_yaml)
        self._assert_config(config, pipeline, model, config_yaml)

    def _assert_config(self, config, pipeline, model, config_yaml):
        self.assertIsNotNone(config)
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(model)
        self._assert_general_config(config_yaml["general"], config)
        self._assert_pipeline(config_yaml["general"], pipeline)
        self._assert_model(config_yaml["model"], model)

    def _assert_model(self, config_yaml_model, model):
        self.assertIsNotNone(model._generator)
        self._assert_generator(config_yaml_model["generator"], model._generator)
        self.assertEqual(OPTIMIZER_TO_OBJ[OPTIMIZER], model.generator_optimizer().__class__)
        self._assert_optimizer_config(config_yaml_model["generator_optimizer_config"],
                         model.generator_optimizer().get_config())

    def _assert_pipeline(self, config_yaml_general, pipeline):
        self.assertEqual(config_yaml_general["train_data_paths"],
                         pipeline.data_path)
        self.assertEqual(config_yaml_general["validation_data_path"],
                         pipeline.validationset_path)
        self._assert_files_in_path(config_yaml_general["test_data_path"],
                                   pipeline.test_img_paths)
        self.assertEqual(config_yaml_general["batch_size"],
                         pipeline.batch_size)
        self._assert_augmentations(AUGMENTATIONS, pipeline.augmentations)

    def _assert_generator(self, config_yaml_generator, generator):
        self.assertEqual(config_yaml_generator["upsample_factor"], generator._upsample_factor)
        self.assertEqual(config_yaml_generator["architecture"], generator._architecture)
        self._assert_loss_functions(LOSS_FUNCTIONS, generator.loss_functions())

    def _assert_general_config(self, config_yaml_general, config):
        self.assertEqual(config_yaml_general["operation"], config.operation)
        self.assertEqual(config_yaml_general["train_data_paths"], config.train_data_paths)
        self.assertEqual(config_yaml_general["validation_data_path"], config.validation_data_path)
        self.assertEqual(config_yaml_general["test_originals_path"], config.test_originals_path)
        self._assert_files_in_path(config_yaml_general["test_data_path"], config.test_data_paths)
        self.assertEqual(config_yaml_general["num_epochs"], config.num_epochs)
        self.assertEqual(config_yaml_general["batch_size"], config.batch_size)
        self.assertEqual(
            RESIZE_FILTER_TO_OBJ[config_yaml_general["resize_filter"]],
            config.resize_filter
        ),
        self.assertEqual(config_yaml_general["crop_imgs"], config.crop_imgs)
        self.assertEqual(config_yaml_general["crop_size"], config.crop_size)
        self.assertEqual(config_yaml_general["num_crops"], config.num_crops)
        self.assertEqual(config_yaml_general["crop_naive"], config.crop_naive)
        self._assert_augmentations(AUGMENTATIONS, config.augmentations)

    def _assert_augmentations(self, aug_strings, aug_funcs):
        self.assertEqual(len(aug_strings), len(aug_funcs))
        self.assertEqual(
            [AUGMENTATION_TO_OBJ[aug_str] for aug_str in aug_strings],
            aug_funcs
        )

    def _assert_optimizer_config(self, config_yaml_optimizer_config, initialized_optimizer_config):
        if config_yaml_optimizer_config is None: return
        for key, val in config_yaml_optimizer_config.items():
            self.assertEqual(
                val, initialized_optimizer_config[key]
            )

    def _assert_loss_functions(self, config_yaml_loss_functions, initialized_loss_functions):
        self.assertEqual(len(config_yaml_loss_functions), len(initialized_loss_functions))
        loss_funcs = [
            {"loss_function": l.__class__.__name__, "weighted": l.weighted, "loss_weight": l.loss_weight}
            for l in initialized_loss_functions
        ]
        self.assertEqual(config_yaml_loss_functions, loss_funcs)

    def _assert_files_in_path(self, paths, files_in_config):
        files = list()
        for path in paths:
            files += [os.path.join(path, fname) for fname in os.listdir(path)]

        self.assertEqual(files, files_in_config)

    def _prepare_train_yaml(self, loss_functions=None):
        loss_funcs = LOSS_FUNCTIONS if loss_functions is None else loss_functions
        return {
            "general": {
                "operation": "training",
                "train_data_paths": TRAIN_DATA_PATHS,
                "validation_data_path": VAL_DATA_PATHS,
                "test_data_path": TEST_DATA_PATH,
                "test_originals_path": TEST_ORIGINALS_PATH,
                "results_save_path": RESULTS_SAVE_PATH,
                "num_epochs": 10,
                "batch_size": 16,
                "resize_filter": RESIZE_FILTER,
                "scale": 4,
                "create_save_dirs": False,
                "train_val_split": 0.5,
                "crop_imgs": True,
                "crop_size": (128, 128, 3),
                "num_crops": 10,
                "crop_naive": True,
                "augmentations": AUGMENTATIONS
            },
            "model": {
                "generator": {
                    "upsample_factor": 4,
                    "architecture": "rrdb",
                    "loss_functions": loss_funcs
                },
                "generator_optimizer": OPTIMIZER,
                "generator_optimizer_config": {
                    "learning_rate": 1e-6,
                    "beta_1": 0.5,
                    "beta_2": 0.25
                }
            }
        }


if __name__ == '__main__':
    unittest.main()
