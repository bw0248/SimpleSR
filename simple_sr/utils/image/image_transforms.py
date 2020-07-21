import numpy as np
import logging
import tensorflow as tf
#import tensorflow_addons as tfa

from simple_sr.utils.image import image_utils
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


DEFAULT_HUE_RANGE = [-0.07, 0.07]
DEFAULT_SATURATION_RANGE = [0.4, 2.0]
DEFAULT_BRIGHTNESS_RANGE = [0.05, 0.2]
DEFAULT_CONTRAST_RANGE = [0.5, 2.5]
DEFAULT_GAMMA_RANGE = [0.5, 1.5]
DEFAULT_JPG_QALITY_RANGE = [10, 50]


def normalize_01(img):
    """
    Normalize img to [0, 1].

    :param img: Tensor with pixel values in [0, 255].
    :return: Tensor withl pixel values normalized to [0, 1].
    """
    return img / 255.0


def normalize_11(img):
    """
    Normalize img to [-1, 1].

    :param img: Tensor with pixel values in [0, 255].
    :return: Tensor with pixel values normalized to [-1, 1].
    """
    return img / 127.5 - 1


def denormalize_11(img):
    """
    Denormalize img from [-1, 1] to [0, 255].

    :param img: Tensor with pixel values in [-1, 1].
    :return: Tensor with pixel values in [0, 255].
    """
    return (img + 1) * 127.5


def crop_naive(img_tensor, num_crops, patch_dims, random_seed=None):
    """
    | Crops a number of patches with specified dimensions from input tensor containing
      elements of `tensorflow.image`.
    | Cropping is considered *naive* because there is no check to verify the
      diversity of cropped patches. This means you will sometimes end up with
      uniform and "uninteresting" batches of cropped patches (e.g. only
      blue patches cropped from the background sky of an image).
    | See `simple_sr.utils.image.image_transforms.crop_divers` for a different approach.
    | Input tensor is not mutated.

    :param img_tensor:
        | Tensor containing elements of `tensorflow.image`.
        | May be of rank 3 or rank 4
    :param num_crops:
        | Number of crops to obtain per element.
    :param patch_dims:
        | List or tuple with Dimensions (including channels) of cropped patches.
        | e.g. to crop 64x64 Patches from an RGB image `patch_dims` should be (64, 64, 3).
    :param random_seed:
        random seed to use for locating and cropping patches, if None cropping will be random.

        .. warning::
            Only set the seed for debugging since every cropped patch will be the same.
    :return:
        Tensor containing patches cropped from input tensor.
    """
    crops = list()
    for _ in range(num_crops):
        crops.append(tf.image.random_crop(img_tensor, patch_dims, random_seed))
    return crops


def crop_divers(img, num_crops, patch_dims, min_variation_patch, min_variation_batch,
                max_trys_patch=100, max_trys_batch=20, random_seed=None):
    """
    | Crops a number of patches with specified dimensions from input tensor containing
      elements of `tensorflow.image`, while trying to maintain a minimum diversity
      inside cropped patches and across batches of patches.

    .. warning::

        This is experimental, performance (in regards to training time) is likely to suffer.
        Some experimentation is needed to find a good balance of efficiency and diversity in patches.

    :param img: A tensorflow.image tensor containing the image to crop patches from.
    :param num_crops: Number of patches to crop.
    :param patch_dims:
        | Dimensions of patches to crop.
        | Channels need to be specified as well, e.g. for 64x64 patches from RGB image => (64, 64, 3).
    :param min_variation_patch: Minimum variation between pixel values of cropped patch for patch
                                to be accepted into batch.
    :param min_variation_batch: Minimum variation between pixel values across batch of cropped patches
                                for batch to be accepted.
    :param max_trys_patch:
        | Number of trys to crop patches above `min_variation_patch` threshold.
        | After number of trys is exhausted every patch will be accepted.
    :param max_trys_batch:
        | Number of trys to assemble batch of cropped patches above `min_variation_batch` threshold.
        | After number of trys is exhausted every batch will be accepted.
    :param random_seed:
        | Random seed for debuggin purposes, every cropped patch will be the same for a single image.
    :return: A tensor containing the specified number of cropped patches
    """
    log.debug(f"cropping {num_crops} sub-images - "
              f"expected variation threshold: {min_variation_patch}")
    num_trys = 0
    while num_trys < max_trys_batch:
        crops = _sample_candidate_crops(img, num_crops, patch_dims, min_variation_patch,
                                        max_trys_patch, random_seed)
        variation_in_batch = tf.math.reduce_std(
            [tf.reduce_mean(img) for img in tf.convert_to_tensor(crops)]
        ).numpy()
        if variation_in_batch > min_variation_batch:
            log.debug(f"found set of crops with enough variation ({variation_in_batch})")
            break
        num_trys += 1
    if num_trys == max_trys_batch:
        log.debug("could not obtain set with enough variation, but maxed out on trys - taking set anyway")
    return tf.convert_to_tensor(crops)


def _sample_candidate_crops(img, num_crops, patch_dims, min_variation_patch,
                            max_trys_patch, random_seed=None):
    number_of_trys = 0
    crops = list()
    while len(crops) < num_crops:
        crop = tf.image.random_crop(img, patch_dims, random_seed)
        variation_in_patch = tf.math.reduce_std(crop).numpy()
        if number_of_trys >= max_trys_patch:
            log.debug(f"already tried {number_of_trys} - taking crop without checking threshold")
            crops.append(crop)
        elif variation_in_patch > min_variation_patch:
            log.debug(f"crop satisfies condition ({variation_in_patch}) - appending to list of crops")
            crops.append(crop)
        else:
            log.debug(f"crop does not satisfy condition - trying again")
        number_of_trys += 1
    return crops

#def rotate(img, angle=None):
#    if angle is None:
#        angle = tf.random.uniform(shape=[], minval=(-35*np.pi/180), maxval=(35*np.pi/180))
#    img_rot = tfa.image.rotate(img, angle, interpolation="BILINEAR")
#    return img_rot


def rotate90(img_tensor, rotations=None):
    """
    | Perform a number of 90 degree rotations on a `tensorflow.image` tensor.
    | Tensor may be of rank 3 or rank 4.
    | Input tensor is not mutated.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :param rotations:
        | Number of rotations to perform.
        | If None number of rotations will be randomly sampled from [1, 3].
    :return:
        Tensor containing rotated versions of input tensor.
    """
    if rotations is None:
        rotations = tf.random.uniform(shape=[], minval=1, maxval=3, dtype=tf.int32)
    return tf.image.rot90(img_tensor, rotations)


def adjust_hue(img_tensor, delta_range=None):
    """
    | Adjust hue value of a tensor containing elements of `tensorflow.image`.
    | The adjusting factor is randomly sampled from the interval specified
      by `delta_range`.
    | Tensor may be of rank 3 or rank 4.
    | Input tensor is not mutated.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :param delta_range:
        | List or tuple specifying lower and upper bounds for sampling interval
          of adjustment delta.
        | If no range is supplied the interval defaults to [-0.07, 0.07].
    :return:
        Tensor containing hue adjusted versions of input tensor.
    """
    if delta_range is None:
        delta_range = DEFAULT_HUE_RANGE
    delta = tf.random.uniform(shape=[], minval=delta_range[0], maxval=delta_range[1])
    adjusted = tf.image.adjust_hue(img_tensor, delta=delta)
    return adjusted


def adjust_saturation(img_tensor, factor_range=None):
    """
    | Adjust saturation value of a tensor containing elements of `tensorflow.image`.
    | The adjusting factor is randomly sampled from the interval specified
      by `factor_range`.
    | Tensor may be of rank 3 or rank 4.
    | Input tensor is not mutated.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :param factor_range:
        | List or tuple specifying lower and upper bounds for sampling interval
          of adjustment factor.
        | If no range is supplied the interval defaults to [0.4, 2.0].
    :return:
        Tensor containing saturation adjusted versions of input tensor.
    """
    if factor_range is None:
        factor_range = DEFAULT_SATURATION_RANGE
    factor = tf.random.uniform(shape=[], minval=factor_range[0], maxval=factor_range[1])
    img_adjusted = tf.image.adjust_saturation(img_tensor, saturation_factor=factor)
    return img_adjusted


def adjust_brightness(img_tensor, delta_range=None):
    """
    | Adjust brightness value of a tensor containing elements of `tensorflow.image`.
    | The adjusting factor is randomly sampled from the interval specified
      by `delta_range`.
    | Tensor may be of rank 3 or rank 4.
    | Input tensor is not mutated.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :param delta_range:
        | List or tuple specifying lower and upper bounds for sampling interval
          of adjustment delta.
        | If no range is supplied the interval defaults to [0.05, 0.2].
    :return:
        Tensor containing brightness adjusted versions of input tensor.
    """
    if delta_range is None:
        delta_range = DEFAULT_BRIGHTNESS_RANGE
    delta = tf.random.uniform(shape=[], minval=delta_range[0], maxval=delta_range[1])
    adjusted = tf.image.adjust_brightness(img_tensor, delta=delta)
    return adjusted


def adjust_contrast(img_tensor, factor_range=None):
    """
    | Adjust contrast value of a tensor containing elements of `tensorflow.image`.
    | The adjusting factor is randomly sampled from the interval specified
      by `factor_range`.
    | Tensor may be of rank 3 or rank 4.
    | Input tensor is not mutated.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :param factor_range:
        | List or tuple specifying lower and upper bounds for sampling interval
          of adjustment factor.
        | If no range is supplied the interval defaults to [0.5, 2.5].
    :return:
        Tensor containing contrast adjusted versions of input tensor.
    """
    if factor_range is None:
        factor_range = DEFAULT_CONTRAST_RANGE
    factor = tf.random.uniform(shape=[], minval=factor_range[0], maxval=factor_range[1])
    adjusted = tf.image.adjust_contrast(img_tensor, factor)
    return adjusted


def adjust_gamma(img_tensor, factor_range=None):
    """
    | Adjust gamma value of a tensor containing elements of `tensorflow.image`.
    | The adjusting factor is randomly sampled from the interval specified
      by `factor_range`.
    | Tensor may be of rank 3 or rank 4.
    | Input tensor is not mutated.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :param factor_range:
        | List or tuple specifying lower and upper bounds for sampling interval
          of adjustment factor.
        | If no range is supplied the interval defaults to [0.5, 1.5].
    :return:
        Tensor containing gamma adjusted versions of input tensor.
    """
    if factor_range is None:
        factor_range = DEFAULT_GAMMA_RANGE
    factor = tf.random.uniform(shape=[], minval=factor_range[0], maxval=factor_range[1])
    return tf.image.adjust_gamma(img_tensor, factor)


def adjust_jpg_quality(img, quality_range=None):
    """
    | Adds jpg noise to a `tensorflow.image` tensor. Tensor needs to be of rank 3.
    | The noise level that will be applied to the input tensor is randomly sampled
      in the interval specified by `quality_range`.
    | Input tensor is not mutated.

    :param img:
        A tensorflow.image.
    :param quality_range:
        | List or tuple for lower and upper bounds for randomly sampling jpg degradation level.
        | Lower values mean more degradation/less jpg quality.
        | If no quality range is supplied interval defaults to [10, 50]

    :return:
        A `tensorflow.image` with jpg degradation
    """
    if quality_range is None:
        quality_range = DEFAULT_JPG_QALITY_RANGE
    quality = tf.random.uniform(
        shape=[], minval=quality_range[0], maxval=quality_range[1], dtype=tf.int32
    )
    return tf.image.adjust_jpeg_quality(img, quality)


def flip_along_x(img_tensor):
    """
    | Flip elements of a tensor along its x-axis. Tensor may be of rank 3 or rank 4.
    | Does not mutate the input tensor.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :return:
        Tensor containing flipped versions of input tensor.
    """
    flipped = tf.image.flip_up_down(img_tensor)
    return flipped


def flip_along_y(img_tensor):
    """
    | Flip elements of a tensor along its y-axis. Tensor may be of rank 3 or rank 4.
    | Does not mutate the input tensor.

    :param img_tensor:
        Tensor containing elements of `tensorflow.image`.
    :return:
        Tensor containing flipped versions of input tensor.
    """
    flipped = tf.image.flip_left_right(img_tensor)
    return flipped


def resize(img_tensor, resize_dims, resize_filter=tf.image.ResizeMethod.BILINEAR, antialias=True):
    """
    | Method to resize `tensorflow.image` tensors to supplied dimensions.
    | Does not mutate the input image.

    :param img_tensor:
        `tensorflow.image` tensor to be resized. May be of rank 3 or rank 4.
    :param resize_dims:
        A tuple of (height, width) to specify the dimensions the input image should be resized to.
    :param resize_filter:
        A `tensorflow.image.ResizeMethod` to use for resizing the input image.
    :param antialias:
        Whether to use antialiasing during resize operation.
    :return:
        A tensor containing resized versions of input tensor.
    """
    return tf.image.resize(
        img_tensor, size=resize_dims,
        method=resize_filter,
        antialias=antialias
    )


def augment_img(img, augmentations=None, return_as_tf_dataset=False):
    """
    Convenience method to create multiple augmented versions of an image.

    :param img:
        `tensorflow.image` tensor to create augmented versions of.
    :param augmentations:
        A list of augmentation functions.
    :param return_as_tf_dataset:
        Whether to return the augmented images as a `tf.dataset` or a python list.
    :return:
        A collection of `tensorflow.image` tensors containing the augmented images
        either inside a list or  a `tf.dataset` depending `return_as_tf_dataset` parameter.
    """
    augmented_imgs = list()
    augmented_imgs.append(img)
    if augmentations is not None:
        for augmentation in augmentations:
            augmented_imgs.append(augmentation(img))
    if return_as_tf_dataset:
        return tf.data.Dataset.from_tensor_slices(augmented_imgs)
    else:
        return augmented_imgs


def get_all_available_augmentations():
    """
    Retrieve all available augmentation functions.

    :return: A list containing all available augmentation functions.
    """
    return [
        #rotate,
        flip_along_x,
        adjust_jpg_quality,
        rotate90,
        adjust_hue, adjust_saturation, adjust_brightness,
        flip_along_y, adjust_contrast,
        adjust_gamma
    ]


if __name__ == "__main__":
    import os

    path = "./data/datasets/reduced/celeba_hq/256"
    imgs = [image_utils.read_img(f"{path}/{fname}") for fname in os.listdir(path)]
    imgs = np.array(imgs)

    #random_crop(imgs[0], crop_size=(128, 128), num_crops=4,
    #            img_dims=(imgs[0].shape[0], imgs[0].shape[1]),
    #            seed=1, record=False)

    kwargs = {
        "r90": rotate90(imgs, 1),
        "r180": rotate90(imgs, 2),
        "r270": rotate90(imgs, 3),
        "rr": rotate90(imgs)
    }

    image_utils.prepare_image_grid(
        "./data", "rotation_test", (256, 256), **kwargs
    )


