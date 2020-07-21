import ruamel.yaml
import tensorflow as tf
import logging

from simple_sr.utils.config import yaml_helper
from simple_sr.utils.models import model_builder
from simple_sr.utils.models.loss_functions.discriminator_loss import DiscriminatorLoss
from simple_sr.utils.models.loss_functions.ra_discriminator_loss import RaDiscriminatorLoss
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class Discriminator:
    """
    | `Discriminator` to train `SRModel` in adversarial mode.
    | The `Discriminators` job is to critique synthetic and real samples.
    | Also calculates its own loss and keeps track of metrics for epochs and batches.

    :param loss_function:
        | Loss function to calculate loss of `Discriminator`, currently available loss functions are
          standard GAN discriminator loss and relativistic average GAN discriminator loss.
        | More info on different types of losses for GANs can be found here: https://arxiv.org/abs/1807.00734.
    :param relativistic:
        | Whether the `Discriminator` is relativistic, if relativistic is true there won't be a final sigmoid layer
          as the last layer of the discriminators architecture.
    :param label_smoothing:
        | Whether to apply label smoothing, this can help to stabilize the training by making the `Discriminators`
          job a little harder.

            * if false target labels for the discriminator will be either 0 or 1 for synthetic and
              real samples respectively
            * if true target labels will in range [0, smoothing_offset] for fake labels
              and [1 - smoothing_offset, 1 + smoothing_offset] for real labels

        For more tweaks to make GANs more stable see this github repo: https://github.com/soumith/ganhacks
    :param smoothing_offset:
        Sets upper and lower bound for random noise in target labels (if label_smoothing is True).
    :param num_filters:
        Number of filters in convolutional layers of discriminators architecture.
    :param alpha:
        Negative slope coefficient for Leaky ReLU activation function.
    :param kernel_size:
        Kernel size in convolutional layers of discriminators architecture.
    :param momentum:
        Momentum for batch normalization.
    :param initializer:
        Initializer for weights initialization of discriminator.
    :param input_dims:
        Dimensions of input images to the discriminator.
    """
    def __init__(self,
                 loss_function,
                 relativistic,
                 label_smoothing=False,
                 smoothing_offset=0.3,
                 num_filters=64,
                 alpha=0.2,
                 kernel_size=3,
                 momentum=0.8,
                 initializer=None,
                 input_dims=(None, None)):
        self._model = model_builder.build_discriminator(
            input_dims=input_dims, num_filters=num_filters, alpha=alpha,
            kernel_size=kernel_size, momentum=momentum,
            relativistic=relativistic, initializer=initializer
        )
        self._relativistic = relativistic
        self._label_smoothing = label_smoothing
        self._smoothing_offset = smoothing_offset
        if not self._label_smoothing:
            self._smoothing_offset = 0.0
        self._loss_function = loss_function

        self._batch_metrics = dict()
        self._epoch_metrics_train = dict()
        self._epoch_metrics_valid = dict()

        for metrics_dict in [self._batch_metrics, self._epoch_metrics_train,
                             self._epoch_metrics_valid]:
            metrics_dict[self._loss_function.name] = tf.keras.metrics.Mean()
            if self._loss_function.weighted:
                metrics_dict[f"weighted_{self._loss_function.name}"] = tf.keras.metrics.Mean()
            if self._loss_function.name == "ra_adversarial_loss":
                metrics_dict["discriminator_accuracy"] = tf.keras.metrics.Mean()
                metrics_dict["SR_accuracy"] = tf.keras.metrics.Mean()
                metrics_dict["HR_accuracy"] = tf.keras.metrics.Mean()
            else:
                metrics_dict["discriminator_accuracy"] = tf.keras.metrics.BinaryAccuracy()
                metrics_dict["SR_accuracy"] = tf.keras.metrics.BinaryAccuracy()
                metrics_dict["HR_accuracy"] = tf.keras.metrics.BinaryAccuracy()

    def model(self):
        """
        Retrieve the Discriminators model.

        :return: Instance of type tf.keras.model.
        """
        return self._model

    def loss_function(self):
        """
        Retrieve loss function of `Discriminator`.

        :return: Initialized loss function object from simple_sr.utils.models.loss_functions module
        """
        return self._loss_function

    def batch_metrics(self):
        return self._batch_metrics

    def epoch_metrics(self, train=True):
        if train:
            return self._epoch_metrics_train
        else:
            return self._epoch_metrics_valid

    def reset_epoch_metrics(self):
        """
        Reset all training and validation metrics.
        """
        for metric in self._epoch_metrics_train.values():
            metric.reset_states()
        for metric in self._epoch_metrics_valid.values():
            metric.reset_states()

    def reset_batch_metrics(self):
        """
        Reset all batch metrics.
        """
        for metric in self._batch_metrics.values():
            metric.reset_states()

    def formatted_epoch_metrics(self, train=True):
        """
        Return formatted string of epoch metrics for logging.

        :param train: Request either training or validation metrics.
        :return: Formatted metrics string of training or validation metrics, depending on supplied parameter.
        """
        if train:
            return self._format_metrics(self._epoch_metrics_train)
        else:
            return self._format_metrics(self._epoch_metrics_valid)

    # Note: safer to have separate methods for training/validation batches, because of TF graph compilation internals
    def critic_train_batch(self, sr_batch, hr_batch):
        """
        Critique synthetic and real data training batches and keep track of metrics.

        :param sr_batch: Batch of synthetic training data.
        :param hr_batch: Batch of real training data.
        :return: Tuple of tensors containing the likelihood of samples being real for
                 sr_batch and hr_batch respectively.
        """
        sr_critic = self._model(sr_batch, training=True)
        hr_critic = self._model(hr_batch, training=True)

        synth_labels = tf.zeros_like(sr_critic)
        real_labels = tf.ones_like(hr_critic)

        self._batch_metrics["discriminator_accuracy"](synth_labels, sr_critic)
        self._batch_metrics["discriminator_accuracy"](real_labels, hr_critic)
        self._batch_metrics["SR_accuracy"](synth_labels, sr_critic)
        self._batch_metrics["HR_accuracy"](real_labels, hr_critic)

        self._epoch_metrics_train["discriminator_accuracy"](synth_labels, sr_critic)
        self._epoch_metrics_train["discriminator_accuracy"](real_labels, hr_critic)
        self._epoch_metrics_train["SR_accuracy"](synth_labels, sr_critic)
        self._epoch_metrics_train["HR_accuracy"](real_labels, hr_critic)

        return sr_critic, hr_critic

    def critic_validation_batch(self, sr_batch, hr_batch):
        """
        Critique synthetic and real data validation batches and keep track of metrics.

        :param sr_batch: Batch of synthetic validation data.
        :param hr_batch: Batch of real validation data.
        :return: Tuple of tensors containing the likelihood of samples being real
                 for sr_batch and hr_batch respectively.
        """
        sr_critic = self._model(sr_batch, training=False)
        hr_critic = self._model(hr_batch, training=False)

        synth_labels = tf.zeros_like(sr_critic)
        real_labels = tf.ones_like(hr_critic)

        self._batch_metrics["discriminator_accuracy"](synth_labels, sr_critic)
        self._batch_metrics["discriminator_accuracy"](real_labels, hr_critic)
        self._batch_metrics["SR_accuracy"](synth_labels, sr_critic)
        self._batch_metrics["HR_accuracy"](real_labels, hr_critic)

        self._epoch_metrics_valid["discriminator_accuracy"](synth_labels, sr_critic)
        self._epoch_metrics_valid["discriminator_accuracy"](real_labels, hr_critic)
        self._epoch_metrics_valid["SR_accuracy"](synth_labels, sr_critic)
        self._epoch_metrics_valid["HR_accuracy"](real_labels, hr_critic)

        return sr_critic, hr_critic

    def calculate_train_loss(self, sr_critic, hr_critic):
        """
        | Delegates calculation of training loss to loss function.
        | Target labels for loss calculation will generated according to parameters
          `label_smoothing` and `smoothing_offset`.

        :param sr_critic: `Discriminators` critique of a synthetic training data batch.
        :param hr_critic: Discriminators` critique of a real training data batch.
        :return: Loss calculated by discriminators loss function.
        """
        sr_labels, hr_labels = self._get_labels(sr_critic, hr_critic)
        return self._loss_function(
            sr_critic, hr_critic, sr_labels, hr_labels, self._batch_metrics,
            self._epoch_metrics_train
        )

    def calculate_validation_loss(self, sr_critic, hr_critic):
        """
        | Delegates calculation of validation loss to loss function.
        | Target labels for loss calculation will generated according to parameters
          `label_smoothing` and `smoothing_offset`.

        :param sr_critic: `Discriminators` critique of a synthetic validation data batch.
        :param hr_critic: `Discriminators` critique of a real validation data batch.
        :return: Loss calculated by discriminators loss function.
        """
        sr_labels, hr_labels = self._get_labels(sr_critic, hr_critic)
        return self._loss_function(
            sr_critic, hr_critic, sr_labels, hr_labels, self._batch_metrics,
            self._epoch_metrics_valid
        )

    def _format_metrics(self, metrics):
        metrics_info = f"\t{self._loss_function.name}: {metrics[self._loss_function.name].result():.5f}\n"
        for name, metric in metrics.items():
            if name != self._loss_function.name:
                metrics_info += f"\t{name}: {metric.result():.5f}\n"
        return metrics_info

    def _get_labels(self, sr_critic, hr_critic):
        noise_sr = 0
        noise_hr = 0

        if self._label_smoothing:
            noise_hr = tf.random.uniform(
                shape=hr_critic.shape, minval=0.0, maxval=0.5, dtype=tf.float64
            )
            noise_sr = tf.random.uniform(
                shape=sr_critic.shape, minval=0.0, maxval=1.0, dtype=tf.float64
            ) * self._smoothing_offset

        sr_labels = tf.zeros_like(sr_critic, dtype=tf.float64) + noise_sr
        hr_labels = tf.ones_like(hr_critic, dtype=tf.float64) - self._smoothing_offset + noise_hr
        return sr_labels, hr_labels

    def __str__(self):
        return "Discriminator\n" \
               f"relativistic: {self._relativistic}\n" \
               f"label smoothing: {self._label_smoothing}\n" \
               f"smoothing offset: {self._smoothing_offset}\n" \
               f"loss function:\n {self._loss_function}\n"

    @staticmethod
    def initialize_relativistic(weighted_loss=False,
                                loss_weight=1.0,
                                num_filters=64,
                                alpha=0.2,
                                kernel_size=3,
                                momentum=0.8,
                                initializer=None,
                                input_dims=(None, None)):
        """
        Convenience method to initialize a relativistic average GAN discriminator with corresponding loss function.

        :param weighted_loss:
            Whether loss function should weighted.
        :param loss_weight:
            Factor for weighted loss.
        :param num_filters:
            Number of filters in convolutional layers of discriminators architecture.
        :param alpha:
            Negative slope coefficient for Leaky ReLU activation function.
        :param kernel_size:
            Kernel size in convolutional layers of discriminators architecture.
        :param momentum:
            Momentum for batch normalization.
        :param initializer:
            Initializer for weights initialization of discriminator.
        :param input_dims:
            Dimensions of input images to the discriminator.
        :return:
            Initialized Discriminator object.
        """
        return Discriminator(
            loss_function=RaDiscriminatorLoss(weighted=weighted_loss, loss_weight=loss_weight),
            relativistic=True,
            num_filters=num_filters,
            alpha=alpha,
            kernel_size=kernel_size,
            momentum=momentum,
            initializer=initializer,
            input_dims=input_dims
        )

    @staticmethod
    def initialize_standard(weighted_loss=False,
                            loss_weight=1.0,
                            label_smoothing=False,
                            smoothing_offset=0.0,
                            num_filters=64,
                            alpha=0.2,
                            kernel_size=3,
                            momentum=0.8,
                            initializer=None,
                            input_dims=(None, None)):
        """
        Convenience method to initialize a standard GAN discriminator with corresponding loss function.

        :param weighted_loss:
            Whether loss function should weighted.
        :param loss_weight:
            Factor for weighted loss.
        :param label_smoothing:
            | Whether to apply label smoothing, this can help to stabilize the training by making the `Discriminators`
              job a little harder.

                * if false target labels for the discriminator will be either 0 or 1 for synthetic and
                  real samples respectively
                * if true target labels will in range [0, smoothing_offset] for fake labels
                  and [1 - smoothing_offset, 1 + smoothing_offset] for real labels

            For more tweaks to make GANs more stable see this github repo: https://github.com/soumith/ganhacks
        :param smoothing_offset:
            Sets upper and lower bound for random noise in target labels.
        :param num_filters:
            Number of filters in convolutional layers of discriminators architecture.
        :param alpha:
            Negative slope coefficient for Leaky ReLU activation function.
        :param kernel_size:
            Kernel size in convolutional layers of discriminators architecture.
        :param momentum:
            Momentum for batch normalization.
        :param initializer:
            Initializer for weights initialization of discriminator.
        :param input_dims:
            Dimensions of input images to the discriminator.
        :return:
            Initialized Discriminator object.
        """
        return Discriminator(
            loss_function=DiscriminatorLoss(weighted=weighted_loss, loss_weight=loss_weight),
            relativistic=False,
            label_smoothing=label_smoothing,
            smoothing_offset=smoothing_offset,
            num_filters=num_filters,
            alpha=alpha,
            kernel_size=kernel_size,
            momentum=momentum,
            initializer=initializer,
            input_dims=input_dims
        )

    @staticmethod
    def from_yaml(config_yaml):
        """
        Initialize discriminator from supplied yaml config.

        :param config_yaml:
            yaml file containing specification for discriminator, see examples for yaml structure
        :return:
            Initalized discriminator object.
        """
        # check whether yaml is already loaded, try to load if not
        if type(config_yaml) is not dict:
            with open(config_yaml) as f:
                conf_yaml = ruamel.yaml.load(f)
        else:
            conf_yaml = config_yaml
        loss_funcs = yaml_helper.init_loss_functions_from_yaml(
            conf_yaml["model"]["discriminator"]
        )
        conf_yaml["model"]["discriminator"]["loss_functions"] = loss_funcs
        return Discriminator(**conf_yaml["model"]["discriminator"])

