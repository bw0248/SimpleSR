import tensorflow as tf
import os
import logging
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.models import Model

from simple_sr.utils.models import model_builder
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class VGGLoss:
    """
    | Loss function to calculate loss for Generator based on differences between
      activations in feature maps of a reference network. The reference network used here
      can be either the VGG19 (default) or VGG16 network.
    | Loss functions of this type are usually called Perceptual Loss and aim to
      drive the Generator towards producing visually more pleasing results, compared
      to pixelwise loss functions.
      See https://arxiv.org/abs/1609.04802 for more info on perceptual loss.

    | After initialization the `VGGLoss` object can be used as a functor for
      loss calculation:

    .. code::

        vgg_loss = VGGLoss(output_layers="block5_conv4",
                           after_activation=False,
                           feature_scale=1.0,
                           loss_weight=1.0,
                           total_variation_loss=False)
        ...
        loss = vgg_loss(hr_batch, sr_batch, hr_critic, sr_critic)

    :param output_layers:
        | Layers of Vgg19 network to compare feature maps of synthesized and real data samples on.
          May be a list of layers, the loss of each layer will be summed up.
        | The names of each layer can be found here: https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py
    :param feature_scale: Scaling factor for each feature map.
    :param loss_weight:
        | Factor to weight the
        | This can be useful if VGG loss is combined with an additional pixel-wise loss,
        | which might be orders of magnitudes higher and could therefor outweigh VGG loss.
    :param total_variation_loss:
        | Whether to use an additional total variation loss as used in some variants of
          SRGAN (https://arxiv.org/abs/1609.04802).
    :param total_varation_weight: Factor to weight total variation loss.
    :param after_activation:
        | Whether calculate loss on feature maps after activation function or before.
        | See ESRGAN paper (https://arxiv.org/abs/1809.00219) for an explanation
        | of the benefits and downsides.
    :param track_metrics: whether the class should update the supplied metrics dictionaries.
    :param vgg16: Whether to use VGG16 instead of VGG19.
    :param custom_weights: Whether to initialize VGG network with custom weights.
    :param custom_weights_path: Path to custom weights file (.h5 file).
    """
    def __init__(self, output_layers, feature_scale=1.0, loss_weight=1.0,
                 total_variation_loss=False, total_varation_weight=2*10e-8,
                 after_activation=True, track_metrics=True,
                 vgg16=False, custom_weights=False, custom_weights_path=None):
        if vgg16:
            self.preprocess_func = tf.keras.applications.vgg16.preprocess_input
        else:
            self.preprocess_func = tf.keras.applications.vgg19.preprocess_input
        self.name = "vgg_loss"
        self.mse = tf.keras.losses.MeanSquaredError()
        self.track_metrics = track_metrics
        self.feature_scale = feature_scale
        self.loss_weight = loss_weight
        self.weighted = False
        if self.loss_weight != 1.0:
            self.weighted = True
        self.total_variation_loss = total_variation_loss
        self.total_variation_weight = total_varation_weight
        self.after_activation = after_activation
        self.loss = 0
        self.weighted_loss = 0

        if self.after_activation:
            log.debug("requested vgg features after activation - building standard vgg network")
            net = VGG19
            if vgg16:
                net = VGG16
            vgg = net(
                input_shape=(None, None, 3), include_top=False, weights="imagenet",
                pooling=None
            )
            if custom_weights:
                if custom_weights_path is None:
                    raise ValueError("no custom weights path supplied")
                if not os.path.isfile(custom_weights_path):
                    raise ValueError("can't locate custom weights")
                vgg.load_weights(custom_weights_path)
        else:
            # build custom vgg to allow loss calculation before activation
            log.debug("requested vgg features before activation - building custom vgg network")
            net = model_builder.build_vgg_19
            if vgg16:
                net = model_builder.build_vgg_16
            vgg = net(input_shape=(None, None, 3), load_custom_weights=custom_weights,
                      custom_weights_path=custom_weights_path)
        vgg.trainable = False

        self.output_layers = output_layers
        if type(self.output_layers) is not list:
            self.output_layers = [self.output_layers]
        outputs = [vgg.get_layer(layer_name).output for layer_name in self.output_layers]
        self.model = Model(inputs=[vgg.input], outputs=outputs)
        log.debug(f"initialized vgg loss - output layers: {self.output_layers}, "
                  f"feature scaling: {self.feature_scale}, "
                  f"loss_weight: {self.loss_weight}, after activation: {self.after_activation}")

    @tf.function
    def __call__(self, hr_batch, sr_batch, hr_critic, sr_critic, batch_metrics, epoch_metrics, denormalize=True):
        """
        Calculate vgg loss for a batch of High-Resolution real data samples and synthesized
        High-Resolution samples.

        .. important::
            Pixel values for VGG need to be in [0, 255]. So you can either supply batches
            with pixels in that range, or if your pixels are in [-1, 1] you can use the
            `denormalize` flag for conversion. Any other combination will not work.

        .. note::
            The parameters `hr_critique` and `sr_critique` will not be used/needed for calculation
            of vgg loss, but the function needs to adhere to the (implicit) Generator
            loss function interface.

        :param hr_batch:
            | Tensor of real data High-Resolution samples.
            | Pixel values either need to be in [0, 255] or [-1, 1] with `denormalize=True`.
        :param sr_batch:
            | Tensor of synthesized High-Resolution samples with equal shape as `hr_batch`.
            | Pixel values either need to be in [0, 255] or [-1, 1] with `denormalize=True`.
        :param hr_critic: Not needed, may be `None`.
        :param sr_critic: Not needed, may be `None`.
        :param batch_metrics: Optional dictionary to store batch metrics.
        :param epoch_metrics: Optional dictionary to store epoch metrics.
        :param denormalize: Whether to denormalize from [-1, 1] to [0, 255].
        :return: (Weighted) vgg loss for batch.
        """
        if denormalize:
            hr_batch = (hr_batch + 1) * 127.5
            sr_batch = (sr_batch + 1) * 127.5
        hr_preprocessed = self.preprocess_func(hr_batch)
        sr_preprocessed = self.preprocess_func(sr_batch)

        hr_features = self.model(hr_preprocessed, training=False)
        self._hr_feats = hr_features
        if type(hr_features) is not list:
            hr_features = [hr_features]
        scaled_hr_features = [hr_feature * self.feature_scale for hr_feature in hr_features]

        sr_features = self.model(sr_preprocessed, training=False)
        self._sr_feats = sr_features
        if type(sr_features) is not list:
            sr_features = [sr_features]
        scaled_sr_features = [sr_feature * self.feature_scale for sr_feature in sr_features]

        self.loss = 0
        for hr_feature, sr_feature in zip(scaled_hr_features, scaled_sr_features):
            self.loss += self.mse(hr_feature, sr_feature) * self.loss_weight

        if self.total_variation_loss:
            self.loss += self.total_variation_weight * tf.reduce_sum(
                tf.image.total_variation(sr_batch)
            )

        # TODO: fix weighted loss tracking
        if self.track_metrics:
            batch_metrics[self.name](self.loss)
            epoch_metrics[self.name](self.loss)
            try:
                batch_metrics[f"weighted_{self.name}"](self.weighted_loss)
                epoch_metrics[f"weighted_{self.name}"](self.weighted_loss)
            except KeyError:
                pass
        return self.loss

    def visualize_feature_maps(self, picture, denormalize=True):
        _picture = picture
        if denormalize:
            _picture = (picture + 1) * 127.5
        preprocessed = preprocess_input(_picture)
        features = self.model(preprocessed, training=False)
        return features

    def __str__(self):
        return f"## Vgg Loss\n" \
               f"output layers: {self.output_layers}\n" \
               f"feature scaling: {self.feature_scale}\n" \
               f"after activation: {self.after_activation}\n" \
               f"loss weight: {self.loss_weight}\n"\
               f"total variation loss: {self.total_variation_loss}\n"\
               f"total variation loss weight: {self.total_variation_weight}\n"


if __name__ == "__main__":
    pass
