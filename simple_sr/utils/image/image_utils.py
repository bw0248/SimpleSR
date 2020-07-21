import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


def read_img(fpath, normalize_func=None, yield_path=False):
    """
    Read image from supplied path into `tensorflow.image` tensor.

    :param fpath: File path to image file.
    :param normalize_func:
        | Normalization function to apply, None means no normalization takes place.
        | Functions to either normalize to [0, 1] or [-1, 1] are in `simple_sr.utils.image.image_transforms`.
    :param yield_path: Whether image file path should be returned as well.
    :return: `tensorflow.image` tensor and optionally the path of the image as well.
    """
    img = tf.io.read_file(fpath)
    img = tf.image.decode_png(img)
    img = tf.cast(img, tf.float32)
    if normalize_func is not None:
        img = normalize_func(img)
    if yield_path:
        return img, fpath
    else:
        return img


def tensor_to_img(tensor):
    """
    Converts a tensor to `PIL.Image` object.

    :param tensor: Tensor to be converted.
    :return: `PIL.Image` from input tensor.
    """
    if tensor.shape.rank == 4 and tensor.shape[0] == 1:
        tensor = _extract_tensor(tensor)
    return tf.keras.preprocessing.image.array_to_img(tensor)


def reconstruct_from_overlapping_patches(patches, image_height, image_width, pixel_overlap,
                                         horizontal_padding, vertical_padding):
    """
    Reconstructs an image from supplied overlapping patches.

    :param patches: Tensor of rank 4 containing the patches that will be stitched together.
    :param image_height: Height of the resulting image.
    :param image_width: Width of the resulting image.
    :param pixel_overlap: Pixel overlap per patch per direction (north, east, south, west) in number of pixels.
    :param horizontal_padding: Number of pixel rows that were appended to the bottom of the image
                               before extraction of patches.
    :param vertical_padding: Number of pixel columns that were appended on the right side of the image
                             before extraction of patches.
    :return: A Tensor of rank 3 contatining the reconstructed image.
    """
    if patches.shape.rank != 4:
        raise ValueError("Tensor with patches needs to be of rank 4")

    _patches = patches[:, pixel_overlap: -pixel_overlap, pixel_overlap: -pixel_overlap, :]
    return _reconstruct(
        _patches, image_height, image_width,
        image_height + horizontal_padding, image_width + vertical_padding)


def reconstruct_from_patches(patches, original_height, original_width, horizontal_padding=0,
                             vertical_padding=0):
    """
    Reconstructs single image from supplied non-overlapping patches.

    :param patches: Tensor of rank 4 containing patches.
    :param original_height: Height of image to be reconstructed.
    :param original_width: Width of image to be reconstructed.
    :param vertical_padding: Number of columns that were appended to the image before extraction of patches.
    :param horizontal_padding: Number of rows that were appended to the image before extraction of patches.
    :return: Tensor of rank 3 containing the combined patches.
    """
    if patches.shape.rank != 4:
        raise ValueError("Tensor with patches needs to be of rank 4")
    if horizontal_padding < 0 or vertical_padding < 0:
        raise ValueError("Padding can't be negative")

    return _reconstruct(patches, original_height, original_width,
                        original_height + horizontal_padding, original_width + vertical_padding)


def segment_into_patches(tensor, patch_width=32, patch_height=32, pixel_overlap=0):
    """
    Segments input tensor into patches for memory efficient inference on large pictures.

    .. note:

        For a typical use-case of:
            1. segment image into patches
            2. upscale patches
            3. reconstruct full image from upscaled patches

        you should consider using `pixel_overlap` > 0 for better results.

    :param tensor: Tensor to segment into patches, needs to be rank 3.
    :param patch_width: Width of extracted patches.
    :param patch_height: Height of extracted patches.
    :param pixel_overlap: Amount of overlap per patch per direction (north, east, south, west) in pixels
    :return: A tuple containing tensor of rank 4 with patches of input tensor,
             and a tensor of shape (2, 2) containing the padding that was used/needed.
             Structure of padding:
             :code:`[[number of padded rows top, number of padded rows bottom],`
             :code:`[number of padded columns left, number of padded columns right]]`
    """
    if tensor.shape.rank != 3 and (tensor.shape.rank == 4 and tensor.shape[0] != 1):
        raise ValueError("Tensor must be of rank 3")

    _tensor = tensor
    if tensor.shape.rank == 4:
        _tensor = _extract_tensor(tensor)

    if _tensor.shape[0] < patch_height or _tensor.shape[1] < patch_width:
        raise ValueError("Patch dimensions are larger than image size")

    if pixel_overlap != 0:
        return _segment_with_overlap(_tensor, patch_width, patch_height, pixel_overlap=pixel_overlap)
    else:
        return _segment(_tensor, patch_width, patch_height)


def _segment_with_overlap(tensor, patch_width, patch_height, pixel_overlap=5):
    # pad to a multiple of patch_width and patch_height and additionally add #{pixel overlap} pixels
    horizontal_padding = [pixel_overlap, pixel_overlap]
    vertical_padding = [pixel_overlap, pixel_overlap]
    if tensor.shape[0] % patch_height != 0:
        horizontal_padding[1] += (patch_height - tensor.shape[0]) % patch_height
    if tensor.shape[1] % patch_width != 0:
        vertical_padding[1] += (patch_width - tensor.shape[1]) % patch_width
    padding = [horizontal_padding, vertical_padding, [0, 0]]

    padded = tf.pad(
        tensor, mode="CONSTANT", paddings=padding, constant_values=0
    )

    patches = list()
    for row in range(pixel_overlap, padded.shape[0] - pixel_overlap, patch_width):
        for col in range(pixel_overlap, padded.shape[1] - pixel_overlap, patch_height):
            x_start = col - pixel_overlap
            x_end = col + patch_width + pixel_overlap
            y_start = row - pixel_overlap
            y_end = row + patch_height + pixel_overlap
            patches.append(
                padded[y_start: y_end, x_start: x_end, :]
            )
    return tf.convert_to_tensor(patches), padding[:-1]


def _segment(tensor, patch_width, patch_height):
    horizontal_padding = [0, 0]
    vertical_padding = [0, 0]
    if tensor.shape[0] % patch_height != 0:
        horizontal_padding = [0, (patch_height - tensor.shape[0]) % patch_height]
    if tensor.shape[1] % patch_width != 0:
        vertical_padding = [0, (patch_width - tensor.shape[1]) % patch_width]
    padding = [horizontal_padding, vertical_padding]

    segments = tf.space_to_batch([tensor], [patch_height, patch_width], padding)
    segments = tf.split(segments, patch_height * patch_width, 0)
    segments = tf.stack(segments, 3)
    segments = tf.reshape(segments, [-1, patch_height, patch_width, 3])
    return segments, padding


def _reconstruct(patches, original_height, original_width, padded_height, padded_width):
    patch_height = patches.shape[1]
    patch_width = patches.shape[2]
    patch_channels = patches.shape[3]

    reconstructed = tf.reshape(
        patches,
        [1, (padded_height//patch_height), (padded_width//patch_width),
         patch_height * patch_width, patch_channels]
    )
    reconstructed = tf.split(reconstructed, patch_height * patch_width, 3)
    reconstructed = tf.stack(reconstructed, 0)
    reconstructed = tf.reshape(
        reconstructed,
        [patch_height * patch_width, (padded_height//patch_height), (padded_width//patch_width), patch_channels]
    )
    reconstructed = tf.batch_to_space(reconstructed, [patch_height, patch_width], [[0, 0], [0, 0]])[0]
    return reconstructed[0: original_height, 0: original_width, :]


def save_single(tensor, save_dir, fname, label=None):
    """
    Converts tensors of rank 3 or rank 4 to `PIL` image objects and saves them to disk.

    :param tensor: Tensor containing the images to be saved.
    :param save_dir: Folder to save images to.
    :param fname: Name of files to save images in (will be suffixed with position
                  in tensor if multiple images are saved)
    :param label: Optional labels, will be shown in the bottom left of the saved image.
    """
    if tensor.shape.rank < 3 or tensor.shape.rank > 4:
        raise ValueError("Tensor must be of rank 3 or rank 4")

    if tensor.shape.rank == 3:
        _save_as_img(tensor, save_dir, fname, label)
    else:
        for idx, t in enumerate(tensor):
            _save_as_img(t, save_dir, f"{fname}_{idx}", label)


def _save_as_img(tensor, save_dir, fname, label=None):
    img = tensor_to_img(tensor)
    if label is not None:
        _annotate_img(img, label, (0, 255, 0))
    os.makedirs(save_dir, exist_ok=True)
    img.save(f"{save_dir}/{fname}.png")


def combine_halfs(left_tensor, right_tensor, left_label, save_dir, fname, right_label="interpolated", grid=False):
    """
    Creates a combined image from two images by stitching together left and right half respectively of
    each supplied image. A typical use case would be comparing an image upscaled with Super-Resolution to
    an image upscaled with interpolation.

    :param left_tensor:
        | Tensor containing elements that will be on the left side of the resulting image.
        | Tensor may be of rank 3 or rank 4.
    :param right_tensor:
        | Tensor containing elements that will be on the right side of the resulting image.
        | Tensor may be of rank 3 or rank 4, must have same dimensions as `left_tensor`.
    :param left_label: String to labels left side of resulting image.
    :param right_label: String to labels right side of resulting image.
    :param save_dir: Path to save resulting image to.
    :param fname: File name to save resulting image under.
    :param grid:
        | Option to additionally plot all stitched images together in a grid.
        | `left_tensor` and `right_tensor` must be of rank 4 to use this option.
    """
    if left_tensor.shape[0] != right_tensor.shape[0]:
        raise ValueError("number of sr and lr images does not match")
    if grid and (left_tensor.shape[0] % 2 != 0 or left_tensor.shape[0] < 4):
        raise ValueError("can only prepare image grid for an even number of at least 4 images")

    imgs = list()
    for idx, (sr, lr) in enumerate(zip(left_tensor, right_tensor)):
        sr_img = tensor_to_img(sr)
        _annotate_img(sr_img, left_label, (0, 255, 0))
        lr_img = tensor_to_img(lr)
        lr_img = lr_img.resize(sr_img.size)
        _annotate_img(lr_img, right_label, (255, 0, 0), loc="right")

        main_img = Image.new("RGB", sr_img.size, (255, 255, 255))

        half = sr_img.width // 2
        end = sr_img.width
        bottom = sr_img.height
        sr_img = sr_img.crop((0, 0, half, bottom))
        lr_img = lr_img.crop((half, 0, end, bottom))

        main_img.paste(sr_img, (0, 0))
        main_img.paste(lr_img, (half, 0))

        draw = ImageDraw.Draw(main_img)
        draw.line((half, 0, half, bottom), fill=128)

        imgs.append(main_img)
        os.makedirs(save_dir, exist_ok=True)
        main_img.save(f"{save_dir}/{fname}_{idx}.png")

    if grid:
        num_rows = len(imgs) // 4
        num_cols = 4
        total_width = 0
        total_height = 0
        for img in imgs:
            total_width += img.width
            total_height += img.height
        grid_width = total_width // num_rows
        grid_height = total_height // num_cols

        grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        x_loc = 0
        y_loc = 0
        for idx, img in enumerate(imgs):
            if idx == num_cols:
                x_loc = 0
                y_loc = grid.height // 2
            grid.paste(img, (x_loc, y_loc))
            x_loc += img.width

        os.makedirs(save_dir, exist_ok=True)
        grid.save(f"{save_dir}/{fname}_grid.png")


def prepare_image_grid(save_dir, fname, low_res_key=None, original=None, psnr=None, ssim=None, **kwargs):
    """
    Prepares and saves an image grid for comparison. Supplied images must have the same dimensions.

    :param save_dir: Path to save image grid to.
    :param fname: File name to save image grid in.
    :param original: Optionally an original image can be plotted next to the image grid
                     (for instance if image in grid are patches from some image).
    :param psnr: Optional dictionary with keys corresponding to keys in :code:`kwargs` and values containing
                 tensors of PSNR values. PSNR values will be annotated in corresponding pictures.
    :param ssim: Optional dictionary with keys corresponding to keys in :code:`kwargs` and values containing
                 tensors of SSIM values. SSIM values will be annotated in corresponding pictures.
    :param low_res_key: Optional key for low-resolution images in :code:`kwargs`.
                        Images corresponding to this key, will be padded and centered to align
                        with larges images in grid.
    :param kwargs: A dictionary containing image tensors as values.
                   Keys will be used as labels strings inside plotted images.
    """
    num_imgs = -1
    for tensor in kwargs.values():
        if num_imgs == -1:
            num_imgs = tensor.shape[0]
        elif tensor.shape[0] != num_imgs:
            raise ValueError("received differing amount of images per supplied model - can't produce grid")

    if psnr is not None:
        _verfify_supplied_metrics(psnr, kwargs)

    if ssim is not None:
        _verfify_supplied_metrics(ssim, kwargs)

    num_rows = len(kwargs)
    num_cols = 0
    max_height = -1
    max_width = -1
    for label, tensors in kwargs.items():
        if label not in ["hr", "ground truth"]:
            num_cols = max(num_cols, tensors.shape[-4])
            max_height = max(max_height, tensors.shape[-3])
            max_width = max(max_width, tensors.shape[-2])

    # TODO: document magic keys in kwargs
    try:
        kwargs["ground truth"] = tf.image.resize(kwargs["ground truth"], size=(max_height, max_width),
                                                 method=tf.image.ResizeMethod.BICUBIC)
    except KeyError:
        pass

    if num_cols == 1:
        # plot images next to each other
        grid_width = num_rows * max_width
        grid_height = num_cols * max_height
        grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
        _labels = list()
        _pics = list()
        _psnr = list()
        _ssim = list()
        for idx, (label, tensor) in enumerate(kwargs.items()):
            _labels.append(label)
            _t = tensor
            if _t.shape.rank == 4:
                _t = tf.reshape(_t, (_t.shape[1:]))
            if label == low_res_key:
                _t = _pad_image(_t, height=max_height, width=max_width)

            _pics.append(_t)
            _psnr.append(psnr[label] if psnr is not None else None)
            _ssim.append(ssim[label] if ssim is not None else None)
        _prepare_img_row(
            _pics, grid, _labels, (0, 255, 0), y_loc=0,
            resize=False, resize_dims=None, psnr_values=_psnr,
            ssim_values=_ssim
        )
    else:
        grid_width = num_cols * max_width
        grid_height = num_rows * max_height

        # account for label on left side if no original is supplied
        if original is None:
            column_label_width = int(grid_width * 0.05)
        else:
            column_label_width = 0

        grid_width += column_label_width

        grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        y_location = 0
        for idx, (label, tensors) in enumerate(kwargs.items()):
            try:
                _psnr = psnr[label]
            except (TypeError, KeyError):
                _psnr = None
            try:
                _ssim = ssim[label]
            except (TypeError, KeyError):
                _ssim = None
            if original is None:
                _annotate_column(
                    grid, label, (0, 255, 0), column_label_width, max_height,
                    ypos=max_height * idx
                )
                labels = None
            else:
                labels = label
            _t = tensors
            if label == low_res_key:
                _t = _pad_image(_t, height=max_height, width=max_width)
            _prepare_img_row(
                _t, grid, labels=labels, color=(0, 255, 0),
                y_loc=y_location, resize=False,
                psnr_values=_psnr, x_axis_offset=column_label_width,
                ssim_values=_ssim
            )
            y_location += max_height

    if original is not None:
        try:
            origin = Image.open(original)
        except AttributeError:
            origin = original
        origin_aspect_ratio = origin.width / origin.height
        original_height = grid.height
        original_width = int(origin_aspect_ratio * original_height)
        original = origin.resize((original_width, original_height))
        _annotate_img(original, "original", (255, 0, 255))

        combined_width = grid_width + original_width
        combined_img = Image.new("RGB", (combined_width, grid.height), (255, 255, 255))
        combined_img.paste(original, (0, 0))
        combined_img.paste(grid, (original_width, 0))

        os.makedirs(save_dir, exist_ok=True)
        combined_img.save(f"{save_dir}/{fname}.png")
    else:
        os.makedirs(save_dir, exist_ok=True)
        grid.save(f"{save_dir}/{fname}.png")


def _pad_image(tensor, height, width):
    if tensor.shape.rank > 4 or tensor.shape.rank < 3:
        raise ValueError("tensor must be of rank 3 or rank 4")
    _tensor = tensor
    if tensor.shape.rank == 3:
        _tensor = tf.reshape(tensor, (1, *tensor.shape))

    horz_pad = (height - tensor.shape[-3]) // 2
    vert_pad = (width - tensor.shape[-2]) // 2
    for idx, t in enumerate(_tensor):
        _t = tf.pad(t, [[horz_pad, horz_pad], [vert_pad, vert_pad], [0, 0]])
        # resize additionally to account for integer calculation differences
        _t = tf.image.resize(_t, size=(height, width))
        _t = tf.reshape(_t, (1, *_t.shape))
        if idx == 0:
            res = _t
        else:
            res = tf.concat([res, _t], axis=0)

    if tensor.shape.rank == 3:
        return res[0]
    return res


def _verfify_supplied_metrics(metrics_dict, img_dict):
    if len(metrics_dict) != len(img_dict.values()):
        raise ValueError("did not receive psnr values for every supplied model result")
    num_psnr_vals = -1
    for psnr_val in metrics_dict.values():
        if num_psnr_vals == -1:
            num_psnr_vals = psnr_val.shape[0]
        elif psnr_val.shape[0] != num_psnr_vals:
            raise ValueError("count of supplied psnr values does not match count of supplied images")


def _annotate_img(img, text, color, loc=None):
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size=(int(max(6, (16 - (1024//img.width))))))
    width, height = font.getsize(text)
    if loc is None:
        loc = (5, img.size[1] - (5 + height))
    elif loc == "right":
        loc = (img.width - (width + 5), img.height - (5 + height))
    elif loc == "ssim":
        loc = (img.width - (width + 5), img.height - (2 * (5 + height)))
    draw.rectangle((*loc, loc[0] + width, loc[1] + height), fill="black")
    draw.text(loc, text, font=font, fill=color)


def _annotate_column(img, text, color, width, height, ypos, xpos=0):
    # draw/annotate horizontal first in tmp image, then rotate and paste to original
    tmp_img = Image.new("RGB", (height, width), (0, 0, 0))
    draw = ImageDraw.Draw(tmp_img)
    font = _load_font(font_size=(int(max(6, (16 - (1024//img.width))))))
    font_width, font_height = font.getsize(text)
    draw.text((5, width - (5 + font_height)), text, font=font, fill=color)
    rot = tmp_img.rotate(90, expand=1)
    img.paste(rot, (xpos, ypos))


def _prepare_img_row(tensors, main_img, labels, color, y_loc, resize=False,
                     resize_dims=None, psnr_values=None, ssim_values=None,
                     x_axis_offset=0):
    _labels = labels
    if type(labels) is not list:
        _labels = [labels] * tensors.shape[0]
    for idx, tensor in enumerate(tensors):
        img = tensor_to_img(tensor)
        if resize:
            img = img.resize(resize_dims)
        if _labels[idx] is not None:
            _annotate_img(img, _labels[idx], color)
        if psnr_values is not None and psnr_values[idx] is not None:
            psnr_value = _extract_metric(psnr_values, idx)
            _annotate_img(img, f"psnr: {psnr_value}", (255, 0, 0), loc="right")
        if ssim_values is not None and ssim_values[idx] is not None:
            ssim_value = _extract_metric(ssim_values, idx)
            _annotate_img(img, f"ssim: {ssim_value}", (255, 0, 0), loc="ssim")
        main_img.paste(img, (x_axis_offset + img.size[0] * idx, y_loc))


def _extract_metric(metric_values, idx):
    if metric_values[idx].numpy() == float("inf"):
        value = u"\u221E"
    elif metric_values[idx].numpy() == -1:
        value = "N/A"
    else:
        try:
            value = f"{metric_values[idx]:.2f}"
        except TypeError:
            value = f"{metric_values[idx][0]:.2f}"
    return value


def _load_font(font_size=10):
    try:
        font = ImageFont.truetype("./resources/NotoSansMono-Bold.ttf", size=font_size)
    except OSError:
        print("cannot locate font, using default font as fallback")
        font = ImageFont.load_default()
    return font


def _extract_tensor(batch):
    t = None
    for tensor in batch:
        t = tensor
    return t


if __name__ == "__main__":
    pass
