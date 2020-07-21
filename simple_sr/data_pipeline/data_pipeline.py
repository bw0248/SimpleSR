import os
import logging
import tensorflow as tf
import sklearn.model_selection

from simple_sr.utils.image import image_transforms, image_utils
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class DataPipeline:
    """
    Data pipeline based on `tensorflow.data` API.
    The high-resolution images from the supplied data path are read into a tf.data dataset, augmented according to
    supplied augmentations and then paired with a downscaled low-resolution version.

    Optionally instead of using an image as whole, a number of patches can be cropped out of each HR sample.
    In regards to efficiency, cropping patches out of a smaller (a few thousand samples) high resolution
    dataset seems to yield better performance.

    One drawback of randomly cropping patches from images, is that there might be batches which contain a lot
    of simple structures that are not very helpful for training. An example would be an image with a blue sky in
    the background and a more complex structure in the foreground. Randomly cropping might yield a batch of just
    simple blue samples cropped from the sky.
    To mitigate this there is an option to validate the diversity in a cropped patch and the diversity across
    a batch of crops. As of now this feature is highly experimental though and will negatively affect efficiency.

    :param hr_img_path:
        Path to high-resolution training images.
    :param scale:
        Resize factor for obtaining low-resolution images from supplied high-resolution images.
    :param resize_filter:
        Resize filter to use for downsampling high-resolution images, defaults to bicubic.
        See `tensorflow.image.ResizeMethod` for available methods.
    :param antialias:
        Whether to use antialiasing during downsampling.
    :param train_val_split:
        Factor to split supplied training images into training and validation set.
        E.g. 0.1 means that 10% of training images will be hold back in validation set.
    :param validationset_path:
        Optional Path to validation data, overrides :code:`validationset_size` -> no splitting will occur.
    :param batch_size:
        Number of samples per batch.
    :param augmentations:
        List of augmentations to perform, see :code:`simple_sr.utils.image.image_transforms`
        for available augmentations.
    :param test_img_paths:
        Path to test image data.
    :param crop:
        Whether patches should be cropped from HR training images, only applies to training and validation sets.
    :param crop_size:
        Tuple of (height, width, channels) to specify dimensions of cropped patches.
    :param num_crops:
        Number of patches to crop for each HR sample.
    :param crop_naive:
        If true cropped patches are always accepted, which might yield batches of very
        similar patches or batches only containing very simple structures.
        If crop_naive is false batches will only be accepted if the diversity is above the
        supplied threshold. This feature is currently experimental and comes with a performance
        penalty in regards to speed.
    :param minimum_variation_patch:
        Threshold for variation inside one patch to be accepted into the batch.
        Only applies if :code:`crop_naive` is False.
    :param minimum_variation_batch:
        Threshold for variation across batch of patches for batch to be accepted.
        Only applies if :code:`crop_naive` is False.
    :param random_seed:
        Random seed for cropping, should only be used for testing as every cropped patch will be the same.
    :param shuffle_buffer_size:
        Size of buffer for `tf.data` shuffling mechanism.
    :param jpg_noise:
        Whether to apply jpg noise to LR samples.
    :param jpg_noise_level:
        JPG noise level, 100 means max jpg degradation, 0 mean no degradation.
    """
    def __init__(self,
                 hr_img_path,
                 scale,
                 resize_filter=None,
                 antialias=True,
                 train_val_split=0.1,
                 validationset_path=None,
                 batch_size=8,
                 augmentations=None,
                 test_img_paths=None,
                 crop=True,
                 crop_size=(80, 80, 3),
                 num_crops=8,
                 crop_naive=True,
                 minimum_variation_patch=0.8,
                 minimum_variation_batch=0.05,
                 random_seed=None,
                 shuffle_buffer_size=4096,
                 jpg_noise=False,
                 jpg_noise_level=50):
        self.scale = scale
        if hr_img_path is None:
            self.data_path = None
        else:
            self.data_path = hr_img_path if type(hr_img_path) is list else [hr_img_path]
        if validationset_path is None:
            self.validationset_path = None
        else:
            self.validationset_path = validationset_path if type(validationset_path) is list \
                                      else [validationset_path]
        self.test_img_paths = test_img_paths
        self.crop = crop

        self.shuffle_buffer_size = shuffle_buffer_size
        self.resize_filter = resize_filter
        if self.resize_filter is None:
            self.resize_filter = tf.image.ResizeMethod.BICUBIC

        self.antialias = antialias
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.crop_naive = crop_naive
        self.minimum_variation_patch = minimum_variation_patch
        self.minimum_variation_batch = minimum_variation_batch
        self.random_seed = random_seed

        self.augmentations = augmentations
        self.jpg_noise = jpg_noise
        if self.jpg_noise and image_transforms.adjust_jpg_quality in self.augmentations:
            log.warning("augmenting hr images with jpg noise "
                        "and additionally degrading lr images with jpg noise")
        self.jpg_noise_level = jpg_noise_level
        self.train_val_split = train_val_split
        self.validationset_size = self.train_val_split

        self.train_imgs, self.val_imgs = self._split_train_val()
        if len(self.train_imgs) > 0:
            self.train_ds = self._prepare_train_set()
        else:
            self.train_ds = None

        if len(self.val_imgs) > 0:
            self.val_ds = self._prepare_val_set()
        else:
            self.val_ds = None

        if self.test_img_paths is not None and len(self.test_img_paths) > 0:
            self.test_imgs = self._prepare_test_set()

    def _split_train_val(self):
        val_list = list()
        img_files = list()
        if self.data_path is not None:
            for data_path in self.data_path:
                img_files += [os.path.join(data_path, fname) for fname in os.listdir(data_path)
                              if os.path.isfile(os.path.join(data_path, fname))]

        if self.validationset_path is not None:
            # separate path for val data was supplied, splitting train data not necessary
            self.validationset_size = 0

        if self.validationset_size <= 0.0 and self.validationset_path is None:
            log.warning("no validation set supplied")

        # split data set
        if self.validationset_path is None and self.data_path is not None and self.validationset_size > 0.0:
            train_list, val_list = sklearn.model_selection.train_test_split(
                img_files, test_size=self.validationset_size,
                random_state=self.random_seed
            )
        # prepare validation set list
        elif self.validationset_path is not None:
            train_list = img_files
            for val_path in self.validationset_path:
                if os.path.isfile(val_path):
                    val_list.append(val_path)
                else:
                    val_list += [os.path.join(val_path, fname) for fname in os.listdir(val_path)
                                 if os.path.isfile(os.path.join(val_path, fname))]
        else:
            train_list = img_files
        return train_list, val_list

    def _prepare_train_set(self):
        train_list_ds = tf.data.Dataset.list_files(self.train_imgs)
        train_hr_ds = train_list_ds.map(
            image_utils.read_img,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_hr_ds = train_hr_ds.cache()

        if self.crop:
            train_hr_ds = train_hr_ds.flat_map(self._crop)

        train_hr_ds = train_hr_ds.flat_map(
            lambda img: image_transforms.augment_img(
                img, self.augmentations, return_as_tf_dataset=True
            )
        )
        train_hr_ds = train_hr_ds.shuffle(self.shuffle_buffer_size)

        if not self.jpg_noise:
            train_hr_ds = train_hr_ds.batch(self.batch_size)

        train_hr_ds = train_hr_ds.map(
            self._prepare_img_pairs,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if self.jpg_noise:
            train_hr_ds = train_hr_ds.batch(self.batch_size)
        return train_hr_ds

    def _prepare_val_set(self):
        val_list_ds = tf.data.Dataset.list_files(self.val_imgs, shuffle=False)
        val_hr_ds = val_list_ds.map(
            image_utils.read_img,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        val_hr_ds = val_hr_ds.cache()
        if self.crop:
            val_hr_ds = val_hr_ds.flat_map(self._crop)
        if not self.jpg_noise:
            val_hr_ds = val_hr_ds.batch(self.batch_size)

        val_hr_ds = val_hr_ds.map(
            self._prepare_img_pairs,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if self.jpg_noise:
            val_hr_ds = val_hr_ds.batch(self.batch_size)
        return val_hr_ds

    def _prepare_test_set(self):
        if type(self.test_img_paths) is not list:
            self.test_img_paths = [self.test_img_paths]

        test_files = list()
        for path in self.test_img_paths:
            if os.path.isfile(path):
                test_files.append(path)
            elif os.path.isdir(path):
                test_files += [os.path.join(path, fname) for fname in os.listdir(path)
                               if os.path.isfile(os.path.join(path, fname))]
            else:
                raise ValueError(f"could not locate path: {path}")

        test_ds = tf.data.Dataset.list_files(test_files, shuffle=False)

        test_ds = test_ds.map(
            lambda fpath: image_utils.read_img(
                fpath, normalize_func=image_transforms.normalize_01, yield_path=True
            )
        )
        return test_ds

    def train_batch_generator(self):
        """ yields a tf.data prefetched dataset containing batched tuples of (lr, hr) training images """
        if self.train_ds is not None:
            return self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            return []

    def validation_batch_generator(self):
        """ yields a tf.data prefetched dataset containing batched tuples of (lr, hr) validation images """
        if self.val_ds is not None:
            return self.val_ds.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            return []

    def test_batch_generator(self, batch_size=8):
        """
        Yields a tf.data dataset containing batched tuples of (test image, test image path).
        If no path for test images was supplied an empty list is returned

        File paths will be used for matching crops with their
        whole original images when evaluating models.

        :param batch_size:
            number of (test image, test image path) per batch
        """
        if self.test_img_paths is not None:
            return self.test_imgs.batch(batch_size)
        else:
            return []

    def _dump_train_dataset(self, save_dir, n_epochs=1):
        for epoch in range(n_epochs):
            for idx, (lr_batch, hr_batch) in enumerate(self.train_batch_generator()):
                image_utils.prepare_image_grid(
                    save_dir, f"epoch{epoch}_{idx}_train", low_res_key="LR",
                    original=None, psnr=None, LR=lr_batch, HR=hr_batch
                )

    def _dump_validation_dataset(self, save_dir, n_epochs=1):
        for epoch in range(n_epochs):
            for idx, (lr_batch, hr_batch) in enumerate(self.validation_batch_generator()):
                image_utils.prepare_image_grid(
                    save_dir, f"epoch{epoch}_{idx}_val", low_res_key="LR",
                    original=None, psnr=None, LR=lr_batch, HR=hr_batch
                )

    def _dump_test_img_file_paths(self):
        batch_gen = self.test_batch_generator()
        for lr_batch, fpath in batch_gen:
            print(fpath.numpy())

    def _crop(self, img):
        if self.crop_naive:
            crops = image_transforms.crop_naive(img, self.num_crops, self.crop_size, self.random_seed)
        else:
            crops = tf.py_function(
                func=image_transforms.crop_divers,
                inp=[img, self.num_crops, self.crop_size, self.minimum_variation_patch, self.minimum_variation_batch],
                Tout=tf.float32
            )
            crops.set_shape((self.num_crops, *self.crop_size))
        return tf.data.Dataset.from_tensor_slices(crops)

    def _prepare_img_pairs(self, hr_img):
        lr_img = hr_img / 255
        hr_img = hr_img / 127.5 - 1
        lr_dims = (
            tf.shape(hr_img)[-3] / self.scale,
            tf.shape(hr_img)[-2] / self.scale
        )
        lr_img = tf.image.resize(lr_img, size=lr_dims, method=self.resize_filter, antialias=self.antialias)
        if self.jpg_noise:
            lr_img = tf.image.adjust_jpeg_quality(
                lr_img, jpeg_quality=(100 - self.jpg_noise_level)
            )
        return lr_img, hr_img

    def __str__(self):
        try:
            augs = [aug_func.__name__ for aug_func in self.augmentations]
            num_augs = len(self.augmentations)
        except TypeError:
            augs = None
            num_augs = 0
        return f"DataPipeline:\n" \
               f"data path: {self.data_path}\n" \
               f"validation data path: {self.validationset_path}\n" \
               f"test images: {self.test_img_paths}\n" \
               f"resize filter: {self.resize_filter}\n" \
               f"antialias: {self.antialias}\n" \
               f"validation set size: {self.validationset_size}\n" \
               f"batch size: {self.batch_size}\n" \
               f"\n# Augmentations\n"\
               f"augmentations: {augs}\n" \
               f"number of augmentations: {num_augs}\n" \
               f"jpg noise: {self.jpg_noise}\n" \
               f"jpg noise level: {self.jpg_noise_level}\n" \
               f"\n# Cropping\n" \
               f"crop: {self.crop}\n" \
               f"crop size: {self.crop_size}\n" \
               f"number of crops per image: {self.num_crops}\n" \
               f"crop naive: {self.crop_naive}\n" \
               f"minimum variation in patch: {self.minimum_variation_patch}\n" \
               f"minimum variation in batch: {self.minimum_variation_batch}\n" \
               f"\n# Various\n" \
               f"random seed: {self.random_seed}\n" \
               f"shuffle buffer size: {self.shuffle_buffer_size}\n" \
               f"\n"

    @staticmethod
    def from_config(config):
        """
        Convenience method to initialize a `DataPipeline` from a config.

        :param config:
            Initialized `ConfigUtil` object from simple_sr.utils.config module
        :return:
            Initialized `DataPipeline` object
        """
        return DataPipeline(
            hr_img_path=config.train_data_paths,
            scale=config.scale, resize_filter=config.resize_filter,
            antialias=config.antialias,
            train_val_split=config.train_val_split,
            validationset_path=config.validation_data_path, batch_size=config.batch_size,
            augmentations=config.augmentations, jpg_noise=config.jpg_noise,
            jpg_noise_level=config.jpg_noise_level,
            test_img_paths=config.test_data_paths,
            crop=config.crop_imgs, crop_size=config.crop_size, random_seed=config.random_seed,
            num_crops=config.num_crops, crop_naive=config.crop_naive,
            shuffle_buffer_size=config.shuffle_buffer_size,
            minimum_variation_patch=config.minimum_variation_patch,
            minimum_variation_batch=config.minimum_variation_batch
        )

    @staticmethod
    def eval_pipeline(config):
        """
        | Convenience method to initialize a `DataPipeline` in evaluation mode.
        | Evaluation mode means that images will be read from supplied paths in config,
          and tuples of (downsampled, ground truth) images will be available
          via :code:`DataPipeline::validation_batch_generator`.

        :param config:
            Initialized `ConfigUtil` object from simple_sr.utils.config module
        :return:
            Initialized `DataPipeline` object with supplied images available in
            validation batch generator
        """
        return DataPipeline(
            hr_img_path=None, scale=config.scale,
            validationset_path=config.test_data_paths,
            batch_size=config.batch_size, resize_filter=config.resize_filter,
            antialias=config.antialias,
            crop=config.crop_imgs, crop_size=config.crop_size, random_seed=config.random_seed,
            num_crops=config.num_crops, crop_naive=config.crop_naive,
            minimum_variation_patch=config.minimum_variation_patch,
            minimum_variation_batch=config.minimum_variation_batch
        )

    @staticmethod
    def inference_pipeline(config):
        """
        | Convenience method to initialize :code:`DataPipeline` in inference mode.
        | Inference mode means that supplied images will be read from config and
        | tuples (image, image_path) will be available in :code:`DataPipeline::test_batch_generator`.

        :param config:
            Initialized `ConfigUtil` object from simple_sr.utils.config module
        :return:
            Initialized `DataPipeline` object with supplied images available in
            test batch generator
        """
        return DataPipeline(
            hr_img_path=None, test_img_paths=config.test_data_paths,
            antialias=config.antialias,
            scale=config.scale
        )


if __name__ == "__main__":
    pass
