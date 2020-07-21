import tensorflow as tf
import logging
import ruamel.yaml

from simple_sr.utils.config import yaml_helper
from simple_sr.utils.models import model_builder
from simple_sr.utils.models.loss_functions.mean_absolute_error import MeanAbsoluteError
from simple_sr.utils.models.loss_functions.mean_squared_error import MeanSquaredError
from simple_sr.utils.models.loss_functions.adversarial_loss import AdversarialLoss
from simple_sr.utils.models.loss_functions.ra_adversarial_loss import RaAdversarialLoss
from simple_sr.utils.models.loss_functions.vgg_loss import VGGLoss
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class Generator:
    """
    | `Generator` for `SRModel`.
    | The `Generators` job is to generate high-resolution images from supplied low-resolution images.
    | Can either be used in adversarial mode in combination with a `Discriminator` or in non-GAN mode without
      adversarial loss.
    | `Generator` keeps also track of metrics for batches and epochs.

    :param upsample_factor:
        | Factor for increase in image resolution.
        | E.g. if upsample_factor = 2 a 64x64 input image will be upsampled to 128x128.
    :param architecture:
        Architecture for generator, currently only SRResnet and RRDB are available,
        for more info see:
            * https://arxiv.org/abs/1609.04802 (SRResnet/SRGAN)
            * https://arxiv.org/abs/1809.00219 (RRDB/ESRGAN)
    :param loss_functions:
        | Loss functions to calculate the generators loss.
        | Available loss functions can be found in :code:`simple_sr.utils.models.loss_functions`.
          An arbitrary amount of loss functions can be combined.
    :param num_blocks:
        Number of residual building blocks for SRResnet/RRDB (see papers for info/architecture of these blocks).
    :param num_filters:
        Number of filters for convolutional layers in generator architecture.
    :param kernel_size:
        Kernel size for convolutional layers in generator architecture.
    :param residual_scaling:
        Residual scaling for shortcut connections (will only take effect in RRDB architecture).
    :param kernel_initializer:
        Weight initializer for convolutional layers.
    :param batch_norm:
        | Whether to apply batch normalization in Generators architecture.
        | This option is only applicable to SRResnet
          (see RRDB/ESRGAN paper, the authors conclude batch normalization might not be helpful at all times).
    :param input_dims:
        | Dimensions of input images to the generator.
        | Since the generator is fully convolutional this may be (None, None) -> generator can handle arbitrary sizes.
    :param pretrained_model_path:
        | Path to a pretrained model, the model will be loaded and training will be resumed.
        | Can also be used for pretraining a model in non-adversarial mode on pixel loss and then
          continue training in adversarial mode with different loss functions.
          (like VGG loss as the authors in SRGAN/ESRGAN do)
    :param pretrained_model:
        Already loaded keras model, same things apply as for 'pretrained_model_path' except that
        the model is expected to be already loaded
    """
    def __init__(self,
                 upsample_factor,
                 architecture,
                 loss_functions,
                 num_blocks=16,
                 num_dense_blocks=3,
                 num_filters=64,
                 num_convs=4,
                 kernel_size=3,
                 residual_scaling=0.2,
                 kernel_initializer=None,
                 batch_norm=False,
                 input_dims=(None, None),
                 pretrained_model_path=None,
                 pretrained_model=None):
        self._architecture = architecture
        self._upsample_factor = upsample_factor

        if loss_functions is None or loss_functions is list and len(loss_functions) == 0:
            raise ValueError("no loss function for generator supplied")
        if type(loss_functions) is not list:
            loss_functions = [loss_functions]

        self._loss_functions = loss_functions

        self._batch_metrics = dict()
        self._epoch_metrics_train = dict()
        self._epoch_metrics_valid = dict()
        for idx, loss_func in enumerate(self._loss_functions):
            try:
                loss_func_name = loss_func.name
            except AttributeError:
                loss_func_name = f"loss_function_{idx}"
            self._batch_metrics[loss_func_name] = tf.keras.metrics.Mean()
            self._epoch_metrics_train[loss_func_name] = tf.keras.metrics.Mean()
            self._epoch_metrics_valid[loss_func_name] = tf.keras.metrics.Mean()
            try:
                is_weighted = loss_func.weighted
            except AttributeError:
                is_weighted = False
            if is_weighted:
                self._batch_metrics[f"weighted_{loss_func_name}"] = tf.keras.metrics.Mean()
                self._epoch_metrics_train[f"weighted_{loss_func_name}"] = tf.keras.metrics.Mean()
                self._epoch_metrics_valid[f"weighted_{loss_func_name}"] = tf.keras.metrics.Mean()

        self._batch_metrics["generator_loss"] = tf.keras.metrics.Mean()
        self._epoch_metrics_train["generator_loss"] = tf.keras.metrics.Mean()
        self._epoch_metrics_valid["generator_loss"] = tf.keras.metrics.Mean()

        self._num_blocks = num_blocks
        self._num_dense_blocks = num_dense_blocks
        self._num_filters = num_filters
        self._num_convs = num_convs
        self._kernel_size = kernel_size
        self._residual_scaling = residual_scaling
        self._kernel_initializer = kernel_initializer
        self._batch_norm = batch_norm
        self._input_dims = input_dims
        self._pretrained_model_path = pretrained_model_path
        self._pretrained_model = pretrained_model

        if self._pretrained_model:
            log.debug("found pretrained model - using as generator")
            self._model = self._pretrained_model
        else:
            log.debug("no pretrained model found - building from scratch or loading if path was supplied")
            self._model = model_builder.build_or_load_generator_model(
                upsample_factor=self._upsample_factor,
                architecture=self._architecture,
                num_blocks=self._num_blocks, num_dense_blocks=self._num_dense_blocks,
                num_filters=self._num_filters, num_convs=self._num_convs,
                kernel_size=self._kernel_size, residual_scaling=self._residual_scaling,
                kernel_initializer=self._kernel_initializer, batch_norm=self._batch_norm,
                input_dims=self._input_dims, pretrained_model_path=self._pretrained_model_path
            )

    def model(self):
        """
        Retrieve the generators model.

        :return: Instance of type tf.keras.model.
        """
        return self._model

    def set_model(self, model):
        self._model = model

    def loss_functions(self):
        """
        Retrieve registered loss functions of generator.

        :return: List of initialized loss function objects from simple_sr.utils.models.loss_functions module.
        """
        return self._loss_functions

    def batch_metrics(self):
        return self._batch_metrics

    def epoch_metrics(self, train=True):
        if train:
            return self._epoch_metrics_train
        else:
            return self._epoch_metrics_valid

    def reset_epoch_metrics(self):
        for metric in self._epoch_metrics_train.values():
            metric.reset_states()
        for metric in self._epoch_metrics_valid.values():
            metric.reset_states()

    def reset_batch_metrics(self):
        for metric in self._batch_metrics.values():
            metric.reset_states()

    def formatted_epoch_metrics(self, train=True):
        """
        Retrieve formatted string of epoch metrics for logging

        :param train: request either train or validation metrics
        :return: formatted metrics string of training or validation metrics, depending on supplied parameter
        """
        if train:
            return self._format_metrics(self._epoch_metrics_train)
        else:
            return self._format_metrics(self._epoch_metrics_valid)

    def generate(self, lr_batch, training=True):
        """
        Generate batch of high-resolution images based on supplied low-resolution training images.

        :param lr_batch:
            Low resolution input images.
        :param training:
            Whether currently training or validating (batch normalization will be off during validation).
        :return:
            Tensor containing upsampled images.
        """
        return self._model(lr_batch, training=training)

    def calculate_train_loss(self, sr_batch, hr_batch, sr_critic, hr_critic):
        """
        | Delegates calculation of loss to loss functions and calculates total training loss.
        | Loss functions will record training loss metrics.

        :param sr_batch:
            Batch of generated high-resolution samples.
        :param hr_batch:
            Batch corresponding high-resolution ground truth samples.
        :param sr_critic:
            Critique of discriminator for synthetic/generated data training samples
            (only applicable if training in GAN mode, otherwise will be None).
        :param hr_critic:
            Critique of discriminator for real data training samples
            (only applicable if training in GAN mode, otherwise will be None).
        :return:
            Integer representing total loss for training batch.
        """
        total_loss = 0
        for loss_func in self._loss_functions:
            total_loss += loss_func(
                hr_batch, sr_batch, hr_critic, sr_critic, self._batch_metrics,
                self._epoch_metrics_train
            )
        self._batch_metrics["generator_loss"](total_loss)
        self._epoch_metrics_train["generator_loss"](total_loss)
        return total_loss

    # Note: safer to have separate methods for training/validation batches, because of TF graph compilation internals
    def calculate_validation_loss(self, sr_batch, hr_batch, sr_critic, hr_critic):
        """
        | Delegates calculation of validation loss to loss functions and calculates total validation loss.
        | Loss functions will record training loss metrics.

        :param sr_batch:
            Batch of generated high-resolution samples.
        :param hr_batch:
            Batch of corresponding high-resolution ground truth samples.
        :param sr_critic:
            Critique of discriminator for synthetic/generated data validation samples
            (only applicable if training in GAN mode, otherwise will be None).
        :param hr_critic:
            Critique of discriminator for real data validation samples
            (only applicable if training in GAN mode, otherwise will be None).
        :return:
            Integer representing total loss for validation batch.
        """
        total_loss = 0
        for loss_func in self._loss_functions:
            total_loss += loss_func(
                hr_batch, sr_batch, hr_critic, sr_critic, self._batch_metrics,
                self._epoch_metrics_valid
            )
        self._batch_metrics["generator_loss"](total_loss)
        self._epoch_metrics_valid["generator_loss"](total_loss)
        return total_loss

    def _format_metrics(self, metrics):
        metrics_info = f"\ttotal loss: {metrics['generator_loss'].result():.5f}\n"
        for name, metric in metrics.items():
            if name != "generator_loss":
                metrics_info += f"\t{name}: {metric.result():.5f}\n"
        return metrics_info

    def __str__(self):
        loss_funcs_info = ""
        for loss_func in self._loss_functions:
            loss_funcs_info += str(loss_func)
        return "# Generator\n"\
               f"architecture: {self._architecture}\n"\
               f"upsample factor: {self._upsample_factor}\n" \
               f"pretrained model: {self._pretrained_model}\n" \
               f"pretrained model path: {self._pretrained_model_path}\n"\
               f"loss functions:\n {loss_funcs_info}\n" \
               f"number of residual blocks: {self._num_blocks}\n" \
               f"number of filters: {self._num_filters}\n"

    @staticmethod
    def srresnet(upsample_factor,
                 loss_function=None,
                 num_blocks=16,
                 num_filters=64,
                 kernel_size=3,
                 batch_norm=True,
                 input_dims=(None, None),
                 pretrained_model_path=None,
                 pretrained_model=None):
        """
        | Convenience method for initializing SRResnet Generator in non-adversarial mode.
        | Default parameters are set according to SRResnet/SRGAN paper (https://arxiv.org/abs/1609.04802).

        :return:
            Initialized `Generator` instance.
        """
        log.debug(f"Setting up srresnet - "
                  f"pretrained model path: {pretrained_model_path}, "
                  f"pretrained_model: {pretrained_model}")
        if loss_function is None:
            loss_function = [MeanSquaredError(weighted=False, loss_weight=1.0)]
        return Generator(
            upsample_factor=upsample_factor,
            architecture="srresnet",
            loss_functions=loss_function,
            num_blocks=num_blocks,
            num_filters=num_filters,
            kernel_size=kernel_size,
            batch_norm=batch_norm,
            input_dims=input_dims,
            pretrained_model_path=pretrained_model_path,
            pretrained_model=pretrained_model
        )

    @staticmethod
    def rrdb(upsample_factor,
             loss_functions=MeanAbsoluteError,
             loss_weight=1.0,
             num_blocks=16,
             num_dense_blocks=3,
             num_filters=64,
             num_convs=4,
             kernel_size=3,
             residual_scaling=0.2,
             kernel_initializer=None,
             batch_norm=False,
             input_dims=(None, None),
             pretrained_model_path=None,
             pretrained_model=None):
        """
        | Convenience method for initializing RRDB Generator in non-adversarial mode.
        | Default parameters are set according to RRDB/ESRGAN paper (https://arxiv.org/abs/1809.00219).

        :return:
            Initialized `Generator` instance.
        """
        log.debug(f"Setting up rrdb - pretrained model path: {pretrained_model_path}"
                  f"pretrained model path: {pretrained_model_path}, "
                  f"pretrained_model: {pretrained_model}")
        weighted = True if loss_weight != 1.0 else False
        return Generator(
            upsample_factor=upsample_factor,
            architecture="rrdb",
            loss_functions=[loss_functions(weighted=weighted, loss_weight=loss_weight)],
            num_blocks=num_blocks,
            num_dense_blocks=num_dense_blocks,
            num_filters=num_filters,
            num_convs=num_convs,
            kernel_size=kernel_size,
            residual_scaling=residual_scaling,
            kernel_initializer=kernel_initializer,
            batch_norm=batch_norm,
            input_dims=input_dims,
            pretrained_model_path=pretrained_model_path,
            pretrained_model=pretrained_model
        )

    @staticmethod
    def srgan_generator(upsample_factor,
                        vgg_loss,
                        vgg_layer,
                        vgg_feature_scaling=(1/12.75),
                        vgg_loss_weight=1.0,
                        adversarial_loss_weight=1e-3,
                        num_blocks=16,
                        num_filters=64,
                        kernel_size=3,
                        batch_norm=True,
                        input_dims=(None, None),
                        pretrained_model_path=None,
                        pretrained_model=None):
        """
        | Convenience method for initializing SRResnet Generator in adversarial mode.
        | Default parameters are set according to SRResnet/SRGAN paper (https://arxiv.org/abs/1609.04802).

        :return:
            Initialized `Generator` instance.
        """
        log.debug(f"Setting up srgan - pretrained model path: {pretrained_model_path}"
                  f"pretrained model path: {pretrained_model_path}, "
                  f"pretrained_model: {pretrained_model}")
        if vgg_loss:
            loss_functions = [
                VGGLoss(vgg_layer, feature_scale=vgg_feature_scaling,
                        loss_weight=vgg_loss_weight, after_activation=True)
            ]
        else:
            loss_functions = [MeanSquaredError(weighted=False, loss_weight=1.0)]
        if adversarial_loss_weight != 1.0:
            loss_functions.append(AdversarialLoss(weighted=True, loss_weight=adversarial_loss_weight))
        else:
            loss_functions.append(AdversarialLoss(weighted=False, loss_weight=1.0))
        return Generator(
            upsample_factor=upsample_factor,
            architecture="srresnet",
            loss_functions=loss_functions,
            num_blocks=num_blocks,
            num_filters=num_filters,
            kernel_size=kernel_size,
            batch_norm=batch_norm,
            input_dims=input_dims,
            pretrained_model_path=pretrained_model_path,
            pretrained_model=pretrained_model
        )

    @staticmethod
    def esrgan_generator(upsample_factor,
                         vgg_layer="block5_conv4",
                         vgg_feature_scaling=1.0,
                         vgg_loss_weight=1.0,
                         adversarial_loss_weight=5e-3,
                         l1_loss_weight=1e-2,
                         num_blocks=16,
                         num_dense_blocks=3,
                         num_filters=64,
                         num_convs=4,
                         kernel_size=3,
                         input_dims=(None, None),
                         pretrained_model_path=None,
                         pretrained_model=None):
        """
        | Convenience method for initializing RRDB Generator in adversarial mode.
        | Default parameters are set according to RRDB/ESRGAN paper (https://arxiv.org/abs/1809.00219).

        :return:
            Initialized `Generator` instance.
        """
        log.debug(f"Setting up esrgan - pretrained model path: {pretrained_model_path}"
                  f"pretrained model path: {pretrained_model_path}, "
                  f"pretrained_model: {pretrained_model}")
        return Generator(
            upsample_factor=upsample_factor,
            architecture="rrdb",
            loss_functions=[
                MeanAbsoluteError(weighted=True, loss_weight=l1_loss_weight),
                RaAdversarialLoss(weighted=True, loss_weight=adversarial_loss_weight),
                VGGLoss(output_layers=vgg_layer, feature_scale=vgg_feature_scaling,
                        loss_weight=vgg_loss_weight, after_activation=False)
            ],
            num_blocks=num_blocks,
            num_dense_blocks=num_dense_blocks,
            num_filters=num_filters,
            num_convs=num_convs,
            kernel_size=kernel_size,
            residual_scaling=0.2,
            kernel_initializer=None,
            batch_norm=False,
            input_dims=input_dims,
            pretrained_model_path=pretrained_model_path,
            pretrained_model=pretrained_model
        )

    @staticmethod
    def from_yaml(config_yaml):
        """
        Initialize generator from supplied yaml config

        :param config_yaml:
            yaml file containing specification for generator, see examples for yaml structure
        :return:
            Initialized `Generator` instance.
        """
        # check whether yaml is already loaded, try to load if not
        if type(config_yaml) is not dict:
            with open(config_yaml) as f:
                conf_yaml = ruamel.yaml.load(f)
        else:
            conf_yaml = config_yaml
        loss_funcs = yaml_helper.init_loss_functions_from_yaml(
            conf_yaml["model"]["generator"]
        )
        conf_yaml["model"]["generator"]["loss_functions"] = loss_funcs
        return Generator(**conf_yaml["model"]["generator"])

