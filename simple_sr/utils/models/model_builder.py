import logging
import os
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, Add, Lambda,\
    Dense, LeakyReLU, ReLU, Flatten, Activation, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


def build_or_load_generator_model(upsample_factor, architecture, num_blocks, num_filters,
                                  kernel_size, residual_scaling, kernel_initializer, batch_norm,
                                  input_dims, num_convs=4, num_dense_blocks=3, pretrained_model_path=None):
    log.debug("loading or building model")
    if pretrained_model_path is not None:
        log.debug("found path for pretrained model - loading model")
        return tf.keras.models.load_model(pretrained_model_path)
    log.debug("no pretrained model path supplied - creating new network")
    if type(architecture) is str and architecture == "rrdb":
        log.debug("creating model with rrdb architecture")
        return build_enhanced_resnet(
            upsample_factor=upsample_factor, num_filters=num_filters, num_rrdb_blocks=num_blocks,
            num_dense_blocks=num_dense_blocks,
            num_convs=num_convs, kernel_size=kernel_size, residual_scaling_factor=residual_scaling,
            input_dims=input_dims
        )
    elif type(architecture) is str and architecture == "srresnet":
        log.debug("creating model with srresnet architecture")
        return build_resnet(
            upsample_factor=upsample_factor, num_filters=num_filters, num_res_blocks=num_blocks,
            input_dims=input_dims, batch_normalization=batch_norm
        )
    elif callable(architecture):
        log.debug("creating model from supplied function")
        return architecture()
    else:
        raise ValueError("architecture not recognized")


def build_enhanced_resnet(upsample_factor=2, num_filters=64, num_rrdb_blocks=16,
                          num_dense_blocks=3, num_convs=4,
                          kernel_size=3, residual_scaling_factor=0.2, input_dims=(None, None)):
    """
    Residual-in-Residual Dense Block Network as specified in ESRGAN paper (https://arxiv.org/abs/1809.00219).

    :param upsample_factor: Factor for increase in resolution, needs to be in [2, 4, 8].
    :param num_filters: Number of filters in convolutional blocks.
    :param num_rrdb_blocks: Number of residual-in-residual dense building blocks.
    :param num_dense_blocks: Number of dense blocks inside rrdb blocks.
    :param num_convs: Number of convolutions inside dense blocks.
    :param kernel_size: Kernel size of filter for convolutions.
    :param residual_scaling_factor: Scaling factor between dense blocks inside rrdb blocks.
    :param input_dims: Tuple of (Height, Width) for input to network
                       - may be (None, None) for fully convolutional architecture.
    :return: Object of type `Keras.model`.
    """
    # TODO: take initializer as param
    initializer = tf.keras.initializers.he_normal()
    initializer.scale = 0.2

    if upsample_factor not in [2, 4, 8]:
        raise ValueError("upsample factor not supported - please choose either 2, 4 or 8")
    x_in = Input(shape=(*input_dims, 3))     # height, width might vary, channels will always be 3

    x = x_skip = _build_conv_layer(x_in, num_filters=num_filters, kernel_size=3,
                                   initializer=initializer)

    x = _build_rrdb_blocks(
        x, num_blocks=num_rrdb_blocks, num_dense_blocks=num_dense_blocks,
        num_convs=num_convs, num_filters=num_filters, kernel_size=kernel_size,
        residual_scaling=residual_scaling_factor, initializer=initializer
    )

    x = _build_conv_layer(x, num_filters=num_filters, kernel_size=kernel_size,
                          initializer=initializer)

    x = Add()([x_skip, x])

    for _ in range(int(math.log(upsample_factor, 2))):
        x = _subpixel_conv_block(
            x, upsample_factor=2, initializer=initializer, activation=False
        )
        x = LeakyReLU(alpha=0.2)(x)

    x = _build_conv_layer(
        x, num_filters=num_filters, kernel_size=kernel_size, initializer=initializer
    )
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(
        3, kernel_size=3, padding="same", strides=(1, 1), kernel_initializer=initializer,
        activation="tanh"
    )(x)

    return Model(inputs=x_in, outputs=x)


def build_resnet(upsample_factor=2, num_filters=64, num_res_blocks=16, momentum=0.8,
                 input_dims=(None, None), batch_normalization=True):
    """
    Resnet as specified in SRGAN paper (https://arxiv.org/abs/1609.04802)

    :param upsample_factor: Factor for increase in resolution, needs to be in [2, 4, 8].
    :param num_filters: Number of filters in convolutional blocks.
    :param num_res_blocks: Number of residual blocks.
    :param momentum: Momentum for batch normalization.
    :param input_dims: Tuple of (Height, Width) for input to network
                       - may be (None, None) for fully convolutional architecture.
    :param batch_normalization: Whether to include batch normalization layers in architecture.
    :return: Object of type `Keras.model`.
    """
    if upsample_factor not in [2, 4, 8]:
        raise ValueError("upsample factor not supported - please choose either 2, 4 or 8")
    x_in = Input(shape=(*input_dims, 3))     # height, width might vary, channels will always be 3

    x = _build_conv_layer(x_in, num_filters=num_filters, kernel_size=9)
    x = x_skip = PReLU(shared_axes=[1, 2])(x)

    x = _build_res_blocks(x, num_filters, num_res_blocks, momentum, batch_normalization=batch_normalization)

    # reference from krasserm uses default momentum (=0.99) for this layer, keep an eye on this
    x = _build_conv_layer(
        x, num_filters=num_filters, batch_normalization=batch_normalization, momentum=momentum
    )
    x = Add()([x, x_skip])

    for _ in range(int(math.log(upsample_factor, 2))):
        x = _subpixel_conv_block(
            x, upsample_factor=2
        )

    x = Conv2D(3, kernel_size=9, padding="same", strides=(1, 1), activation="tanh")(x)
    return Model(inputs=x_in, outputs=x)


def build_discriminator(input_dims=(None, None), num_filters=64, alpha=0.2, kernel_size=3, momentum=0.8,
                        relativistic=False, initializer=None):
    """
    Discriminator for SRGAN/ESRGAN as specified in both papers.

    :param input_dims: Tuple of (Height, Width) for input images to discriminator.
    :param num_filters: Number of filters in convolutional blocks.
    :param alpha: Alpha value for LeakyReLU activation function.
    :param kernel_size: Kernel size of filter for convolutions.
    :param momentum: Momentum for batch normalization.
    :param relativistic: | Whether discriminator will be used in a standard GAN setting,
                           or in a relativistic GAN setting.
                         | When :code:`relativistic` is True, no Sigmoid layer will be
                           added as the last layer of the network.
    :param initializer: | Initializer for network layers.
                        | If none `He normal` initialization with a scale of 0.2 will be used.
    :return: Object of type `Keras.model`.
    """
    if initializer is None:
        initializer = tf.keras.initializers.he_normal()
        initializer.scale = 0.2

    x_in = Input(shape=(*input_dims, 3))

    x = _build_conv_layer(
        x_in, num_filters=num_filters, kernel_size=kernel_size, strides=(1, 1),
        batch_normalization=False, initializer=initializer
    )
    x = LeakyReLU(alpha=alpha)(x)

    x = _build_conv_layer(
        x, num_filters=num_filters, kernel_size=kernel_size, strides=(2, 2),
        batch_normalization=True, momentum=momentum, initializer=initializer
    )
    x = LeakyReLU(alpha=alpha)(x)

    x = _build_conv_block(
        x, num_filters=num_filters*2, alpha=alpha, kernel_size=kernel_size,
        momentum=momentum, initializer=initializer
    )

    x = _build_conv_block(
        x, num_filters=num_filters*4, alpha=alpha, kernel_size=kernel_size,
        momentum=momentum, initializer=initializer
    )

    x = _build_conv_block(
        x, num_filters=num_filters*8, alpha=alpha, kernel_size=kernel_size,
        momentum=momentum, initializer=initializer
    )

    # TODO: check if Flatten() is actually needed, currently prevents dynamic/varying input size
    x = Flatten()(x)

    x = Dense(1024, kernel_initializer=initializer)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Dense(1, kernel_initializer=initializer)(x)
    if not relativistic:
        log.debug("discriminator is non-relativistic - adding sigmoid layer")
        x = Activation("sigmoid")(x)

    return Model(inputs=x_in, outputs=x)


def build_vgg_19(input_shape=(None, None), load_custom_weights=False, custom_weights_path=None):
    """
    Builds a copy of VGG19. This way it is possible to obtain feature maps before activation layers,
    whereas in the original VGG19 the activations are baked into the convolutional layers and therefore
    can't be accessed.

    :param input_shape: Tuple of (Height, Width) for input to network
                       - may be (None, None) for fully convolutional architecture.
    :param load_custom_weights: Initialize network with custom weights, for instance from a network trained
    on face recognition.
    :param custom_weights_path: Path to custom weights file (.h5 file).
    :return: Object of type `Keras.model`.
    """
    original_vgg = tf.keras.applications.vgg19.VGG19(
        input_shape=(*input_shape, 3), include_top=False, weights="imagenet", pooling=None
    )
    if load_custom_weights:
        if custom_weights_path is None:
            raise ValueError("no path for custom weights supplied")
        if not os.path.isfile(custom_weights_path):
            raise ValueError("can't locate custom weights in supplied path")
        original_vgg.load_weights(custom_weights_path)

    x_in = Input((*input_shape, 3))
    return _custom_vgg(x_in, original_vgg)


def build_vgg_16(input_shape, load_custom_weights=False, custom_weights_path=None):
    """
    Builds a copy of VGG16. This way it is possible to obtain feature maps before activation layers,
    whereas in the original VGG16 the activations are baked into the convolutional layers and therefore
    can't be accessed.

    :param input_shape: Tuple of (Height, Width) for input to network
                       - may be (None, None) for fully convolutional architecture.
    :param load_custom_weights: Initialize network with custom weights, for instance from a network trained
    on face recognition.
    :param custom_weights_path: Path to custom weights file (.h5 file).
    :return: Object of type `Keras.model`.
    """
    original_vgg = tf.keras.applications.vgg16.VGG16(
        input_shape=(*input_shape, 3), include_top=False, pooling=None
    )
    if load_custom_weights:
        if custom_weights_path is None:
            raise ValueError("no path for custom weights supplied")
        if not os.path.isfile(custom_weights_path):
            raise ValueError("can't locate custom weights in supplied path")
        original_vgg.load_weights(custom_weights_path)

    x_in = Input((*input_shape, 3))
    return _custom_vgg(x_in, original_vgg)


def _custom_vgg(x_in, original_vgg):
    x = x_in
    for layer in original_vgg.layers[1:]:
        if "conv" in layer.name:
            weights, bias = layer.get_weights()
            x = Conv2D(
                filters=layer.filters, kernel_size=layer.kernel_size, padding=layer.padding,
                name=layer.name, kernel_initializer=tf.keras.initializers.Constant(weights),
                bias_initializer=tf.keras.initializers.Constant(bias)
            )(x)
            x = ReLU()(x)
        elif "pool" in layer.name:
            x = MaxPooling2D(
                pool_size=layer.pool_size, strides=layer.strides, name=layer.name
            )(x)
        else:
            raise ValueError("layer not recognized")
    return Model(inputs=x_in, outputs=x)


def _subpixel_conv_block(x_in, upsample_factor, initializer=None, activation=True):
    num_filters = int(x_in.shape[3])
    x = _build_conv_layer(x_in, num_filters=num_filters * upsample_factor ** 2,
                          initializer=initializer)
    x = tf.nn.depth_to_space(x, block_size=upsample_factor)
    if activation:
        return PReLU(shared_axes=[1, 2])(x)
    return x


def _build_conv_layer(x_in, num_filters=64, kernel_size=3, padding="same",
                      strides=(1, 1), batch_normalization=False, momentum=0.8, initializer=None):
    x = Conv2D(
        filters=num_filters, kernel_size=kernel_size, padding=padding, strides=strides,
        kernel_initializer=initializer
    )(x_in)
    if batch_normalization:
        x = BatchNormalization(momentum=momentum)(x)
    return x


def _build_conv_block(x_in, num_filters, alpha=0.2, kernel_size=3, momentum=0.8, initializer=None):
    x = _build_conv_layer(
        x_in, num_filters=num_filters, kernel_size=kernel_size, strides=(1, 1),
        batch_normalization=True, momentum=momentum, initializer=initializer
    )
    x = LeakyReLU(alpha=alpha)(x)
    x = _build_conv_layer(
        x, num_filters=num_filters, kernel_size=kernel_size, strides=(2, 2),
        batch_normalization=True, momentum=momentum, initializer=initializer
    )
    return LeakyReLU(alpha=alpha)(x)


def _res_block(x_in, num_filters, momentum, batch_normalization=True):
    """ standard res block of srgan """
    x = _build_conv_layer(
        x_in, num_filters=num_filters, batch_normalization=batch_normalization, momentum=momentum
    )
    x = PReLU(shared_axes=[1, 2])(x)
    x = _build_conv_layer(
        x, num_filters=num_filters, batch_normalization=batch_normalization, momentum=momentum
    )
    x = Add()([x_in, x])    # sum result of res block with skip connection
    return x


def _build_res_blocks(x_in, num_filters, num_res_blocks, momentum, batch_normalization=True):
    for _ in range(num_res_blocks):
        x_in = _res_block(x_in, num_filters, momentum, batch_normalization=batch_normalization)
    return x_in


def _dense_block(x_in, num_convs, num_filters, initializer, kernel_size):
    """ dense block of esrgan """
    prev_layers = [x_in]
    x = x_in
    for _ in range(num_convs):
        x = _build_conv_layer(x, num_filters=(num_filters//2), kernel_size=kernel_size,
                              initializer=initializer)
        x = LeakyReLU(alpha=0.2)(x)
        prev_layers.append(x)
        if len(prev_layers) > 1:
            x = Concatenate(axis=3)(prev_layers[:])     # workaround for https://github.com/tensorflow/tensorflow/issues/30355

    x_out = _build_conv_layer(x, num_filters=num_filters, kernel_size=3, initializer=initializer)
    return x_out


def _rrdb_block(x_in, num_dense_blocks, num_convs, num_filters, kernel_size, residual_scaling, initializer):
    """ residual in residual dense block of esrgan """
    res_connection = x_in
    for _ in range(num_dense_blocks):
        block = _dense_block(res_connection, num_convs, num_filters, initializer=initializer, kernel_size=kernel_size)
        scaled_block = Lambda(lambda x: x * residual_scaling)(block)
        res_connection = Add()([res_connection, scaled_block])
    return res_connection


def _build_rrdb_blocks(x_in, num_blocks, num_dense_blocks, num_convs, num_filters, kernel_size, residual_scaling,
                       initializer):
    block = x_in
    for _ in range(num_blocks):
        block = _rrdb_block(
            x_in=block, num_dense_blocks=num_dense_blocks,num_convs=num_convs,
            num_filters=num_filters, kernel_size=kernel_size,
            residual_scaling=residual_scaling, initializer=initializer
        )
    scaled_unit = Lambda(lambda x: x * residual_scaling)(block)
    res_connection = Add()([x_in, scaled_unit])
    return res_connection


if __name__ == "__main__":
    pass
