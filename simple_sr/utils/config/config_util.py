import sys
import os
import logging
import copy
import time
from pathlib import Path
import tensorflow as tf
from datetime import datetime

from simple_sr.data_pipeline.data_pipeline import DataPipeline
from simple_sr.models.sr_model import SRModel
from simple_sr.models.generator import Generator
from simple_sr.models.discriminator import Discriminator
from simple_sr.utils.config import yaml_helper
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)
AVAILABLE_OPERATIONS = ["training", "evaluation", "inference", "experiment"]


class ConfigUtil:
    """
    | Helper class that encapsulates configuration options, prepares save folders as well as
      Tensorflow summary writers and initializes logger.
    | Depending on the chosen operation a folder structure will be initialized in the supplied save path.

    .. code::

        +- timestamp
            +-- checkpoints (Tensorflow checkpoints)
            +-- logs (Tensorflow summary writer logs)
            +-- models (saved Keras models)
            +-- pics (saved pictures from different sets)
                +-- test (saved pictures from test set)
                +-- train (saved pictures from training set)
                +-- val (saved pictures from validation set)
            |-- config_logfile (dump of `ConfigUtil`, `DataPipeline`, `SRModel` configuration)
            |-- log (log file for debug messages)
            |-- result_logfile (training and validation metrics will be logged here during training)

    | For evaluation there are multiple options how the upsampled images should be saved.
      `save_single`, `grid` and `combine_halfs` can be combined and images will be produces for each option.
      Options `interpolate` and `with_original` define how the image grid should look like.

    :param operation: One of ["training, "evaluation"].
    :param num_epochs: Number of epochs for training.
    :param batch_size: Number of samples per batch.
    :param train_data_paths: Path to High-Resolution training images.
    :param validation_data_path: Path to High-Resolution validation images.
    :param test_data_paths: Path to Low-Resolution test images.
    :param test_originals_path:
        | Optional Path to Originals of test images.
        | If your test images were cropped from some image, you can display the cropped patches
          next to the original whole image in an image grid (see Results Gallery inside documentation
          for examples).
        | File name of original file needs to match the folder name that contains the crops.
          See data folder for examples and structure.
        | Only takes effect in evaluation mode, not during training.
    :param results_save_path: Path where results are saved.
    :param train_val_split: Ration to split Training data into Training, Validation.
        Only applicable if no `validation_data_path` was supplied.
    :param scale: Upsample factor.
    :param resize_filter: Resize Filter for upsampling.
    :param crop_imgs: Whether to crop patches from High-Resolution images, or process complete images.
    :param crop_size:
        | Tuple specifying dimensions of patches to crop.
        | Channels need to be specified as well, e.g. for 64x64 patches from RGB picture => (64, 64, 3)
    :param num_crops: Number of patches to crop from each High-Resolution training image.
    :param crop_naive:
        | Whether to crop naive (without verifying sufficient diversity in cropped patches).
        | See `simple_sr.utils.image.image_transforms.crop_naive` for more info.
    :param minimum_variation_patch: Minimum variation in cropped patch (only applicable if `crop_naive` == False).
    :param minimum_variation_batch: Minimum variation across cropped batch (only applicable if `crop_naive` == False).
    :param augmentations:
        | Augmentation functions to perform during pre-processing.
        | See `simple_sr.utils.image.image_transforms` for available augmentation functions.
    :param jpg_noise:
        | Whether corrupt Low-Resolution samples with jpg noise during preprocessing.
        | This is not to be confused with augmenting High-Resolution images with jpg noise,
          since only Low-Resolution images will be degraded with jpg noise. The idea is that
          the networks learns to smooth out noise.
    :param jpg_noise_level:
        | Degradation level for jpg noise, in interval [0, 100].
        | 100 means max degradation, 0 means no degradation at all.
    :param dry_run: Whether to actually create save dirs.
    :param model_path: Path to model for evaluation.
    :param shuffle_buffer_size: Size of tensorflow shuffle buffer in `DataPipeline`.
    :param random_seed: Random seed for debugging purposes.
    :param early_stop_metric: Metrics `SRModel` should track to determine whether training should be stopped early.
    :param early_stop_patience: How many epochs may go by without improving early stop metric.
    :param save_single:
        | Whether to save upsampled images as single files
        | Only takes effect in evaluation mode
    :param grid: Whether a grid for comparison of models/images should be produced.
    :param interpolate:
        | Whether saved image grids should include a row for interpolated images.
        | Only takes effect in evaluation mode.
    :param with_original:
        | Whether images grids should include the original image.
        | Only takes effect in evaluation mode.
    :param combine_halfs:
        | Whether images should be stitched together for comparison. See `simple_sr.utils.image.image_utils`
          for more info.
        | Only takes effect in evaluation mode.
    """
    def __init__(self, operation, num_epochs, batch_size,
                 train_data_paths, validation_data_path, test_data_paths,
                 test_originals_path, results_save_path, train_val_split,
                 scale, resize_filter,
                 crop_imgs, crop_size, num_crops, crop_naive,
                 minimum_variation_patch, minimum_variation_batch,
                 augmentations, jpg_noise, jpg_noise_level,
                 dry_run, model_path=None, antialias=True,
                 shuffle_buffer_size=4096, random_seed=None,
                 early_stop_metric="psnr", early_stop_patience=5,
                 save_single=False, grid=False, interpolate=False,
                 with_original=False, combine_halfs=False):
        self.dry_run = dry_run
        self.random_seed = random_seed
        if operation not in AVAILABLE_OPERATIONS:
            raise ValueError(f"operation not recognized - choose one of {AVAILABLE_OPERATIONS}")
        self.operation = operation
        if self.operation == "testing":
            # use unix timestamp because more configs are create during testing (prevents errors)
            self.save_dir_name = str(time.time())
        else:
            self.save_dir_name = datetime.now().strftime('%Y%m%d-%H%M%S')

        self.save_figures = True
        self.show_figures = False

        self.train_data_paths = train_data_paths
        if self.train_data_paths:
            if type(self.train_data_paths) is not list:
                self.train_data_paths = [self.train_data_paths]

        self.save_path = os.path.join(results_save_path, self.operation, self.save_dir_name)
        self.validation_data_path = validation_data_path

        self.test_data_paths = test_data_paths
        self.test_originals_path = test_originals_path
        if self.test_data_paths is not None:
            self.test_originals = self._find_originals()

        self.model_path = model_path

        self._validate_data_dirs()

        self.batch_size = batch_size
        self.scale = scale
        self.save_single = save_single
        self.grid = grid
        self.interpolate = interpolate
        self.with_original = with_original
        self.combine_halfs = combine_halfs
        self.antialias = antialias
        if self.operation not in ["inference"]:
            self.num_epochs = num_epochs

            self.crop_imgs = crop_imgs
            self.crop_size = crop_size
            self.num_crops = num_crops
            self.crop_naive = crop_naive
            self.minimum_variation_patch = minimum_variation_patch
            self.minimum_variation_batch = minimum_variation_batch
            self.shuffle_buffer_size = shuffle_buffer_size
            self.resize_filter = resize_filter
            self.train_val_split = train_val_split
            self.augmentations = augmentations
            self.jpg_noise = jpg_noise
            self.jpg_noise_level = jpg_noise_level

            self.early_stop_metric = early_stop_metric
            self.early_stop_patience = early_stop_patience
        else:
            self.hr_pic_size = None
            self.lr_pic_size = None

        self._prepare_save_dirs()
        logger.setup_logger(f"{self.save_path}")

    def base_save_path(self):
        return Path(self.save_path).parent

    def update_config(self, **kwargs):
        for field, value in kwargs.items():
            setattr(self, field, value)

    def reinitialize_save_dirs(self):
        self._prepare_save_dirs()

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s += f"{key} -> {val}\n"
        return s

    def __deepcopy__(self, memodict):
        cls = self.__class__
        copied = cls.__new__(cls)
        memodict[id(self)] = copied
        for key, val in self.__dict__.items():
            if "summary_writer" not in key:
                setattr(copied, key, copy.deepcopy(val, memodict))
        return copied

    @staticmethod
    def training_config(train_data_paths, num_epochs, batch_size, scale,
                        operation="training", validation_data_path=None, test_data_path=None,
                        test_originals_path=None,
                        results_save_path="./",
                        create_save_dirs=True,
                        train_val_split=0.1,
                        crop_imgs=True, crop_size=(96, 96, 3), num_crops=16,
                        crop_naive=True, minimum_variation_patch=0.15,
                        minimum_variation_batch=0.05,
                        augmentations=None,
                        jpg_noise=False, jpg_noise_level=50,
                        shuffle_buffer_size=4096, random_seed=None, resize_filter=None,
                        antialias=True,
                        early_stop_metric="psnr", early_stop_patience=5):
        """
        Convenience method to initialize `ConfigUtil` in training mode.

        :return: Initialized `ConfigUtil` object.
        """
        test_data_paths = ConfigUtil._extract_multiple_data_paths(test_data_path)
        return ConfigUtil(
            train_data_paths=train_data_paths, num_epochs=num_epochs, batch_size=batch_size,
            resize_filter=resize_filter, antialias=antialias,
            scale=scale, operation=operation,
            validation_data_path=validation_data_path, test_data_paths=test_data_paths,
            test_originals_path=test_originals_path,
            dry_run=not create_save_dirs,
            train_val_split=train_val_split,
            crop_imgs=crop_imgs, crop_size=crop_size, num_crops=num_crops,
            crop_naive=crop_naive,
            minimum_variation_patch=minimum_variation_patch,
            minimum_variation_batch=minimum_variation_batch,
            augmentations=augmentations,
            jpg_noise=jpg_noise,
            jpg_noise_level=jpg_noise_level,
            shuffle_buffer_size=shuffle_buffer_size, random_seed=random_seed,
            early_stop_metric=early_stop_metric, early_stop_patience=early_stop_patience,
            results_save_path=results_save_path
        )

    @staticmethod
    def evaluation_config(data_paths, test_originals_path, model_paths, results_save_path,
                          scale=2, batch_size=8,
                          resize_filter=None, antialias=True,
                          crop_imgs=False,
                          crop_size=(128, 128, 3), random_seed=None,
                          num_crops=16, crop_naive=True, minimum_variation_patch=0.15,
                          minimum_variation_batch=0.05,
                          create_save_dirs=True, operation="evaluation",
                          save_single=True, grid=False, interpolate=False,
                          with_original=False, combine_halfs=False):
        """
        Convenience method to initialize `ConfigUtil` in evaluation mode.

        :return: Initialized `ConfigUtil` object.
        """
        _data_paths = ConfigUtil._extract_multiple_data_paths(data_paths)
        return ConfigUtil(
            operation=operation, num_epochs=None, batch_size=batch_size,
            train_data_paths=None, validation_data_path=None, test_data_paths=_data_paths,
            test_originals_path=test_originals_path,
            results_save_path=results_save_path, model_path=model_paths,
            train_val_split=None,
            crop_imgs=crop_imgs, crop_size=crop_size, num_crops=num_crops,
            crop_naive=crop_naive,
            minimum_variation_patch=minimum_variation_patch,
            minimum_variation_batch=minimum_variation_batch,
            scale=scale, resize_filter=resize_filter, antialias=antialias,
            augmentations=[], jpg_noise=None, jpg_noise_level=None,
            dry_run=not create_save_dirs, random_seed=random_seed,
            save_single=save_single, grid=grid, interpolate=interpolate,
            with_original=with_original, combine_halfs=combine_halfs
        )

    @staticmethod
    def from_yaml(config_yaml_path):
        """
        | Convenience method to initialize every component needed for training.
        | Components that are initialized are `ConfigUtil`, `DataPipeline` and `SRModel`.
        | To get an idea of the yaml structure and possible options, look into examples/complex_example.yaml.

        :param config_yaml_path: Path to yaml configuration file.
        :return: Initialized `ConfigUtil`, `DataPipeline` and `SRModel` ready for training.
        """
        conf_yaml = yaml_helper.load_yaml(config_yaml_path)

        if conf_yaml["general"]["operation"] == "training":
            conf_yaml = yaml_helper.prepare_for_training_config(conf_yaml)
            conf = ConfigUtil.training_config(**conf_yaml["general"])
        elif conf_yaml["general"]["operation"] == "evaluation":
            conf_yaml = yaml_helper.prepare_for_evaluation_config(conf_yaml)
            conf = ConfigUtil.evaluation_config(**conf_yaml["general"])
            pipeline = DataPipeline.eval_pipeline(conf)
            return conf, pipeline
        elif conf_yaml["general"]["operation"] == "inference":
            conf = ConfigUtil.evaluation_config(**conf_yaml["general"])
            pipeline = DataPipeline.inference_pipeline(conf)
            return conf, pipeline
        else:
            raise ValueError(f"Operation {conf_yaml['general']['operation']} not supported")

        pipeline = DataPipeline.from_config(conf)

        generator = Generator.from_yaml(conf_yaml)
        generator_optimizer = yaml_helper.string_to_lib_object(
            "tensorflow",
            ["keras", "optimizers", conf_yaml["model"]["generator_optimizer"]]
        )

        generator_optimizer_config = None
        if "generator_optimizer_config" in conf_yaml["model"]:
            generator_optimizer_config = conf_yaml["model"]["generator_optimizer_config"]

        discriminator = None
        discriminator_optimizer = None
        discriminator_optimizer_config = None
        if "discriminator" in conf_yaml["model"]:
            discriminator = Discriminator.from_yaml(conf_yaml)
            discriminator_optimizer = yaml_helper.string_to_lib_object(
                "tensorflow",
                ["keras", "optimizers", conf_yaml["model"]["discriminator_optimizer"]]
            )
            if "discriminator_optimizer_config" in conf_yaml["model"]:
                discriminator_optimizer_config = conf_yaml["model"]["discriminator_optimizer_config"]

        sr_model = SRModel.init(
            conf, generator, generator_optimizer, generator_optimizer_config,
            discriminator, discriminator_optimizer, discriminator_optimizer_config
        )
        return conf, pipeline, sr_model

    def _prepare_save_dirs(self):
        self.perf_logfile = os.path.join(self.save_path, "perf_logfile")
        self.result_logfile = os.path.join(self.save_path, "result_logfile")
        self.final_result = os.path.join(self.save_path, "result")
        self.tf_logfile = os.path.join(self.save_path, "tf_log")
        self.config_logfile = os.path.join(self.save_path, "config_logfile")
        if not self.dry_run:
            os.makedirs(self.save_path, exist_ok=True)
        if not self.operation == "tuning":
            self._add_save_dir("pics", "pic_dir")

        # create folders that are only needed for training or benchmark
        if self.operation not in ["tuning", "testing", "evaluation", "inference"]:
            self._add_save_dir("checkpoints", "checkpoint_dir")
            self._add_save_dir(os.path.join("pics", "test"), "pic_dir_test")
            self._add_save_dir("models", "model_dir")
            self._add_save_dir(os.path.join("pics", "train"), "pic_dir_train")
            self._add_save_dir(os.path.join("pics", "val"), "pic_dir_val")

            # prepare Tensorboard summary writers and corresponding folders
            self._add_save_dir(os.path.join("logs", "train", "epoch"), "log_dir_train_epoch")
            self._add_save_dir(os.path.join("logs", "train", "batch"), "log_dir_train_batch")
            self._add_save_dir(os.path.join("logs", "val", "epoch"), "log_dir_val_epoch")
            self._add_save_dir(os.path.join("logs", "val", "batch"), "log_dir_val_batch")
            self.epoch_train_summary_writer = tf.summary.create_file_writer(self.log_dir_train_epoch)
            self.batch_train_summary_writer = tf.summary.create_file_writer(self.log_dir_train_batch)
            self.epoch_validation_summary_writer = tf.summary.create_file_writer(self.log_dir_val_epoch)
            self.batch_validation_summary_writer = tf.summary.create_file_writer(self.log_dir_val_batch)

    #    if self.operation == "testing":
    #        self.train_summary_writer_resnet = None
    #        self.train_summary_writer_gan = None
    #        self.validation_summary_writer_resnet = None
    #        self.validation_summary_writer_gan = None

    def _add_save_dir(self, dir_name, attribute_name):
        path = os.path.join(self.save_path, dir_name)
        if not self.dry_run:
            os.makedirs(path)
        setattr(self, attribute_name, path)

    def _validate_data_dirs(self):
        if not self.dry_run and self.operation not in ["evaluation", "inference"]:
            ConfigUtil._validate_data_dir(self.train_data_paths)
        if not self.dry_run and self.validation_data_path is not None:
            ConfigUtil._validate_data_dir(self.validation_data_path)
        if not self.dry_run and self.test_data_paths is not None:
            ConfigUtil._validate_data_dir(self.test_data_paths)

    def _find_originals(self):
        # try to find corresponding originals
        if not self.test_originals_path:
            return None
        if not os.path.isdir(self.test_originals_path):
            log.debug("could not locate originals folder")
            return None
        test_data_folders_names = [Path(test_path).stem if os.path.isdir(test_path)
                                   else Path(test_path).parent.name
                                   for test_path in self.test_data_paths]
        test_originals = {fname.split(".")[0]: os.path.join(self.test_originals_path, fname)
                          for fname in os.listdir(self.test_originals_path)
                          if fname.split(".")[0] in test_data_folders_names}
        return test_originals

    @staticmethod
    def _validate_data_dir(data_path):
        if type(data_path) is not list:
            data_path = [data_path]
        for path in data_path:
            if not os.path.isdir(path) and not os.path.isfile(path):
                raise ValueError(
                    f"could not locate dataset - {path} does not exist"
                )

    @staticmethod
    def _extract_multiple_data_paths(test_data_path):
        if test_data_path is None:
            return None
        if type(test_data_path) is not list and os.path.isfile(test_data_path):
            return test_data_path
        if type(test_data_path) is not list:
            test_data_path = [test_data_path]

        test_data_paths = list()
        for path in test_data_path:
            if os.path.isfile(path):
                test_data_paths.append(path)
            else:
                test_data_paths += [os.path.join(path, folder) for folder in
                                    os.listdir(path)
                                    if os.path.isdir(os.path.join(path, folder))
                                    or os.path.isfile(os.path.join(path, folder))]
        return test_data_paths


if __name__ == "__main__":
    pass
