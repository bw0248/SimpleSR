import tensorflow as tf


def psnr(tensor1, tensor2, max_val=2.0):
    """
    Calculates Peak signal-to-noise ratio (PSNR) between images.

    :param tensor1: tensor containing images, can be either of rank 3 or rank 4
    :param tensor2: tensor with the same shape as `tensor1`
    :param max_val: maximal possible pixel value in images, for 8-bit RGB this would be 255, if images are
           normalized to [0,1] the maximal value would be 1.0
           For images in range [-1, 1], this value should be 2.0.
    :return: Tensor of rank 1 containing the resulting PSNR values.
    """
    return tf.image.psnr(tensor1, tensor2, max_val=max_val)


def psnr_on_y(tensor1, tensor2, max_val=2.0):
    """
    Calculates Peak signal-to-noise ration (PSNR) between images on Y-channel.

    :param tensor1: Tensor of rank 3 or rank 4.
    :param tensor2: Tensor with same shape as :code:`tensor1`.
    :param max_val: Maximal possible pixel value in images, for 8-bit RGB this would be 255, if images are
           normalized to [0,1] the maximal value would be 1.0.
           For images in range [-1, 1], this value should be 2.0.
    :return: Tensor of rank 1 containing the resulting PSNR values.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("tensors need to have the same shape")
    if tensor1.shape.rank > 4 or tensor1.shape.rank < 3:
        raise ValueError("tensors need to be either of rank 4 or rank 3")

    tensor_1_y = tf.image.rgb_to_yuv(tensor1)
    tensor_2_y = tf.image.rgb_to_yuv(tensor2)
    if tensor1.shape.rank == 4:
        tensor_1_y = tensor_1_y[:, :, :, 0]
        tensor_2_y = tensor_2_y[:, :, :, 0]
    else:
        tensor_1_y = tensor_1_y[:, :, 0]
        tensor_2_y = tensor_2_y[:, :, 0]

    tensor_1_y = tf.reshape(tensor_1_y, (*tensor_1_y.shape, 1))
    tensor_2_y = tf.reshape(tensor_2_y, (*tensor_2_y.shape, 1))
    return psnr(tensor_1_y, tensor_2_y, max_val=max_val)


def ssim(tensor1, tensor2, max_val=2.0):
    """
    Calculates structural similarity (SSIM) between images.

    :param tensor1: Tensor containing images, tensor can be either rank 3 or rank 4.
    :param tensor2: Tensor containing images with the same shape as :code:`tensor1`.
    :param max_val: Maximal possible pixel value in images, for 8-bit RGB this would be 255, if images are
           normalized to [0,1] the maximal value would be 1.0.
           For images in range [-1, 1], this value should be 2.0.
    :return: Tensor of rank 1 containing the resulting SSIM values.
    """
    return tf.image.ssim(tensor1, tensor2, max_val=max_val)


if __name__ == "__main__":
    pass


