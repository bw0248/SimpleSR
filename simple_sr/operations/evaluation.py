import sys
import os
import logging
from pathlib import Path
import tensorflow as tf

from simple_sr.data_pipeline.data_pipeline import DataPipeline
from simple_sr.utils.image import image_transforms, metrics, image_utils
from simple_sr.utils import logger

log = logging.getLogger(logger.RESULTS_LOGGER)


def evaluate_on_validationdata(config, model_name="", pipeline=None, model=None,
                               save_grid=False, combine_halfs=False, save_single=False,
                               save_prefix="", calc_stats=False):
    """
    Evaluates supplied model on evaluation set of supplied data-pipeline and
    plots resulting distributions in regards to PSNR and SSIM.

    :param config:
        | Initialized object of type :code:`ConfigUtil`.
        | Will be used for save path of resulting plots and to load model/pipeline in case they are not supplied.
    :param model_name: Name of model, will be used as label for plots.
    :param pipeline:
        | Initialized object of type :code:`simple_sr.DataPipeline`.
        | Will be initialized from :code:`config` if None.
    :param model: Object of type `Keras.model`, will be loaded from model path in config if None.
    :param save_grid: Whether so save an image grid.
                          (useful if image grid contains patches)
    :param combine_halfs: Whether to save an image that's upscaled with supplied models on one half and bicubic
                          on the other half.
    :param save_single: Whether to save single images.
    :param save_prefix: Prefix for filenames.
    :param calc_stats: Whether to calculate psnr and ssim.
    """
    if pipeline is None:
        pipeline = DataPipeline.from_config(config)
    if model is None:
        if config.model_path is None:
            raise ValueError("No model was supplied and config does not contain path to model")
        models = {Path(path).stem: _load_model(path) for path in config.model_path}
    else:
        models = model if type(model) is dict else {model_name: model}

    validation_batch_generator = pipeline.validation_batch_generator()

    ground_truth_key = "GT"
    low_res_key = "LR"
    interpolated_key = pipeline.resize_filter
    psnr_y_key = "psnr-y"

    result_pics = dict()
    metrics_res = dict()
    for model_name in models.keys():
        metrics_res[model_name] = {
            "psnr": tf.constant([], dtype=tf.float32),
            psnr_y_key: tf.constant([], dtype=tf.float32),
            "ssim": tf.constant([], dtype=tf.float32)
        }
    metrics_res[interpolated_key] = {
        "psnr": tf.constant([], dtype=tf.float32),
        psnr_y_key: tf.constant([], dtype=tf.float32),
        "ssim": tf.constant([], dtype=tf.float32)
    }

    for idx, (lr_batch, hr_batch) in enumerate(validation_batch_generator):
        # hr_batch, sr_batch are normalized to [-1, 1], whereas lr_batch is normalized to [0, 1]
        # -> need to adjust lr_batch for comparison
        result_pics[ground_truth_key] = hr_batch
        result_pics[low_res_key] = lr_batch

        _lr_batch = lr_batch * 255
        _lr_batch = image_transforms.normalize_11(_lr_batch)
        interpolated = image_transforms.resize(
            _lr_batch, (_lr_batch.shape[-3] * config.scale, _lr_batch.shape[-2] * config.scale),
            resize_filter=pipeline.resize_filter
        )

        # adjust hr image slightly to account for integer calculation of resizing
        hr_batch = image_transforms.resize(
            hr_batch, (interpolated.shape[-3], interpolated.shape[-2])
        )

        if calc_stats:
            metrics_res[interpolated_key]["psnr"] = tf.concat(
                [metrics_res[interpolated_key]["psnr"],
                 metrics.psnr(hr_batch, interpolated, max_val=2.0)],    # mav_val needs to be 2 for imgs in range [-1, 1]
                axis=0
            )
            metrics_res[interpolated_key]["ssim"] = tf.concat(
                [metrics_res[interpolated_key]["ssim"],
                 metrics.ssim(hr_batch, interpolated, max_val=2.0)],
                axis=0
            )

            metrics_res[interpolated_key][psnr_y_key] = tf.concat(
                [metrics_res[interpolated_key][psnr_y_key],
                 metrics.psnr_on_y(hr_batch, interpolated, max_val=2.0)],
                axis=0
            )

        if save_single:
            image_utils.save_single(
                interpolated, os.path.join(config.pic_dir, "interpolated"),
                f"{save_prefix}{idx}"
            )

            image_utils.save_single(
                _lr_batch, os.path.join(config.pic_dir, "low_res"),
                f"{save_prefix}{idx}"
            )

        result_pics[interpolated_key] = interpolated

        segmented = False
        segmented_batch = lr_batch
        batch_height, batch_width = _get_tensor_height_width(lr_batch)
        if _eligible_efficient_inference(lr_batch):
            segmented = True
            pixel_overlap = 32
            segmented_batch, padding = image_utils.segment_into_patches(
                lr_batch, patch_width=128, patch_height=128,
                pixel_overlap=pixel_overlap
            )
            segmented_batch = tf.convert_to_tensor(segmented_batch)
        for model_name, model in models.items():
            sr_batch = _upscale(model, segmented_batch)
            if segmented:
                scaled_pixel_overlap = pixel_overlap * config.scale
                sr_batch = image_utils.reconstruct_from_overlapping_patches(
                    sr_batch, image_height=(batch_height * config.scale),
                    image_width=(batch_width * config.scale), pixel_overlap=scaled_pixel_overlap,
                    horizontal_padding=padding[0][1] * config.scale - scaled_pixel_overlap,
                    vertical_padding=padding[1][1] * config.scale - scaled_pixel_overlap
                )
                if sr_batch.shape.rank == 3:
                    sr_batch = tf.reshape(sr_batch, (1, *sr_batch.shape))
            result_pics[model_name] = sr_batch

            if calc_stats:
                metrics_res[model_name]["psnr"] = tf.concat(
                    [metrics_res[model_name]["psnr"],
                     metrics.psnr(hr_batch, sr_batch, max_val=2.0)],
                    axis=0
                )

                metrics_res[model_name][psnr_y_key] = tf.concat(
                    [metrics_res[model_name][psnr_y_key],
                     metrics.psnr_on_y(hr_batch, sr_batch, max_val=2.0)],
                    axis=0
                )

                metrics_res[model_name]["ssim"] = tf.concat(
                    [metrics_res[model_name]["ssim"],
                     metrics.ssim(hr_batch, sr_batch, max_val=2.0)],
                    axis=0
                )

            if save_single:
                image_utils.save_single(
                    sr_batch, os.path.join(config.pic_dir, model_name, "single"),
                    f"{save_prefix}{idx}"
                )

            if combine_halfs:
                image_utils.combine_halfs(
                    left_tensor=sr_batch,
                    right_tensor=image_transforms.resize(lr_batch, _get_tensor_height_width(sr_batch)),
                    left_label=model_name, right_label=interpolated_key,
                    save_dir=os.path.join(config.pic_dir, model_name, "half"),
                    fname=f"{save_prefix}{idx}"
                )

        # extract psnr values of last processed batch
        if calc_stats:
            #batch_psnr = {name: _metrics["psnr"][idx * config.batch_size:]
            batch_psnr = {name: _metrics["psnr"][idx * config.batch_size:]
                          for name, _metrics in metrics_res.items()}
            batch_psnr[ground_truth_key] = tf.constant([float("inf")] * result_pics["Ground truth"].shape[0])
            batch_psnr[low_res_key] = tf.constant([-1] * result_pics["Low-Resolution"].shape[0])

            batch_ssim = {name: _metrics["ssim"][idx * config.batch_size:]
                          for name, _metrics in metrics_res.items()}
            batch_ssim[ground_truth_key] = tf.constant([1.0] * result_pics["Ground truth"].shape[0])
            batch_ssim[low_res_key] = tf.constant([-1] * result_pics["Low-Resolution"].shape[0])

        else:
            batch_psnr = None
            batch_ssim = None

        if save_grid:
            image_utils.prepare_image_grid(
                save_dir=os.path.join(config.pic_dir, "grids"),
                fname=f"{save_prefix}{idx}",
                low_res_key=low_res_key,
                psnr=batch_psnr, ssim=batch_ssim,
                **result_pics

            )

    if calc_stats:
        for model_name, result in metrics_res.items():
            if model_name not in [interpolated_key, ground_truth_key, low_res_key]:
                log.info(f"Average PSNR for {model_name}: {tf.reduce_mean(result['psnr'])}")
                log.info(f"Average PSNR on y-channel for {model_name}: {tf.reduce_mean(result[psnr_y_key])}")
                log.info(f"Average SSIM for {model_name}: {tf.reduce_mean(result['ssim'])}")
                log.info("")
        log.info(f"Average PSNR for {interpolated_key}: {tf.reduce_mean(metrics_res[interpolated_key]['psnr'])}")
        log.info(f"Average PSNR on y-channel for {interpolated_key}: {tf.reduce_mean(metrics_res[interpolated_key][psnr_y_key])}")
        log.info(f"Average SSIM for {interpolated_key}: {tf.reduce_mean(metrics_res[interpolated_key]['ssim'])}")


def evaluate_on_testdata(config, save_single=True, grid=False, interpolate=False,
                         with_original=False, combine_halfs=False, pipeline=None,
                         models=None, save_prefix="", segmentation_min_width=1000,
                         segmentation_min_height=1000):
    """
    Evaluates model(s) supplied via config on test data set of supplied data-pipeline.
    Images will be plotted for each supplied model according to supplied paramaters.
    :code:`save_single`, :code:`grid` and :code:`combine_halfs` can be combined and images will be produced
    and stored for each option. :code:`interpolate` and :code:`with_original` are
    options to define the appearance of :code:`grid`.


    :param config:
        | Initialized object of type :code:`ConfigUtil` from :code:`simple_sr.utils.config` module.
        | Models will be loaded from :code:`model_paths` in config object.
    :param save_single: Whether to save upscaled images as single images.
    :param grid: Whether so save an image grid. Supplied models will be plotted along rows, images across rows.
    :param interpolate: Whether to add a row with interpolated images in grid for comparision.
    :param with_original: Whether to plot original image next to image grid
                          (useful if image grid contains patches)
    :param combine_halfs: Whether to save an image that's upscaled with supplied models on one half and bicubic
                          on the other half.
    :param pipeline:
        | Initialized object of :code:`DataPipeline` from :code:`simple_sr.data_pipeline` package.
        | Will be initialized from config if None.
    :param models: Dict of name: model where model is of type Keras.model.
        | If no dict is supplied, models will be load from path in `config.model_path`.
    :param save_prefix: Prefix for filenames.
    :param segmentation_min_width: Minimal width of images to be segmentated for memory efficient inference.
    :param segmentation_min_height: Minimal height of images to be segmented for memory efficient inference.
    """
    if pipeline is None:
        pipeline = DataPipeline.from_config(config)
    if models is None:
        models = {Path(path).stem: _load_model(path) for path in config.model_path}
    results = dict()
    test_batch_generator = pipeline.test_batch_generator(config.batch_size)
    for idx, (lr_batch, file_path) in enumerate(test_batch_generator):
        original_name = Path(file_path.numpy()[0].decode("utf-8")).parent.stem
        segmented = False
        _lr_batch = lr_batch
        batch_height, batch_width = _get_tensor_height_width(_lr_batch)
        if _eligible_efficient_inference(_lr_batch, min_width=segmentation_min_width,
                                         min_height=segmentation_min_height):
            segmented = True
            pixel_overlap = 32
            _lr_batch, padding = image_utils.segment_into_patches(
                _lr_batch, patch_width=128, patch_height=128,
                pixel_overlap=pixel_overlap
            )
            _lr_batch = tf.convert_to_tensor(_lr_batch)
        for name, model in models.items():
            sr_batch = _upscale(model, _lr_batch)
            if segmented:
                scaled_pixel_overlap = pixel_overlap * config.scale
                sr_batch = image_utils.reconstruct_from_overlapping_patches(
                    sr_batch, image_height=(batch_height * config.scale),
                    image_width=(batch_width * config.scale), pixel_overlap=scaled_pixel_overlap,
                    horizontal_padding=padding[0][1] * config.scale - scaled_pixel_overlap,
                    vertical_padding=padding[1][1] * config.scale - scaled_pixel_overlap
                )
                if sr_batch.shape.rank == 3:
                    sr_batch = tf.reshape(sr_batch, (1, *sr_batch.shape))
            results[name] = sr_batch

            if save_single:
                image_utils.save_single(
                    sr_batch, os.path.join(config.pic_dir, original_name, "single"),
                    f"{save_prefix}{idx}_{original_name}_{name}"
                )

            if combine_halfs:
                image_utils.combine_halfs(
                    left_tensor=sr_batch,
                    right_tensor=image_transforms.resize(lr_batch, (_get_tensor_height_width(sr_batch))),
                    left_label=name, right_label="interpolated",
                    save_dir=os.path.join(config.pic_dir, original_name, "half"),
                    fname=f"{save_prefix}{idx}_{original_name}_{name}"
                )

        if interpolate:
            results["interpolated"] = image_transforms.resize(
                lr_batch, (_get_tensor_height_width(sr_batch)),
                resize_filter=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )

        if save_single:
            for img in results["interpolated"]:
                image_utils.save_single(
                    img, os.path.join(config.pic_dir, "interpolated"),
                    f"{save_prefix}{idx}"
                )

        original = None
        if with_original:
            try:
                original = config.test_originals[original_name]
            except KeyError:
                original = None

        if grid:
            image_utils.prepare_image_grid(
                save_dir=os.path.join(config.pic_dir, "grids"),
                fname=f"{save_prefix}{idx}_{original_name}", low_res_key=None,
                psnr=None, original=original, **results
            )


def _load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except OSError:
        print(f"Error could not locate model at path: {model_path}, exiting")
        sys.exit(1)
    return model


def _get_tensor_height_width(tensor):
    if tensor.shape.rank == 4:
        return tensor.shape[1], tensor.shape[2]
    elif tensor.shape.rank == 3:
        return tensor.shape[0], tensor.shape[1]
    else:
        raise ValueError(f"Received tensor with unexpected rank: {tensor.shape.rank}")


def _eligible_efficient_inference(tensor, min_width=1000, min_height=1000):
    if tensor.shape.rank != 3 and tensor.shape.rank != 4:
        return False
    if tensor.shape.rank == 4 and tensor.shape[0] != 1:
        return False
    batch_width, batch_height = _get_tensor_height_width(tensor)
    if batch_width > min_width and batch_height > min_height:
        return True
    return False


def _upscale(model, lr_batch):
    _sr_batch = list()
    _lr_batch = lr_batch
    if _lr_batch.shape.rank == 4:
        _lr_batch = tf.reshape(_lr_batch, (_lr_batch.shape[0], 1, *_lr_batch.shape[1:]))
    for batch in _lr_batch:
        _sr_batch.append(model(batch, training=False))
    _sr_batch = tf.convert_to_tensor(_sr_batch)
    return tf.reshape(_sr_batch, (-1, *_sr_batch.shape[2:]))


if __name__ == "__main__":
    pass
