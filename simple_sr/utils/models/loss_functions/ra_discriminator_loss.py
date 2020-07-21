import tensorflow as tf
import logging
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class RaDiscriminatorLoss:
    """
    | Relativistic average loss function for Discriminator in relativistic GAN setting.
    | After initialization the `RaDiscriminatorLoss` object can be used as
      a functor to calculate loss of the Discriminator:

    .. code::

        ra_loss = RaDiscriminatorLoss()
        ...
        loss = ra_loss(hr_batch, sr_batch, hr_critic, sr_critic)

    | If `track_metrics` is True, supplied metrics dictionaries will updated with calculated loss.

    :param weighted: whether loss should be weighted
    :param loss_weight: weight factor for loss
    :param track_metrics: whether the class should update the supplied metrics dictionaries.
    """
    def __init__(self, weighted=False, loss_weight=1.0, track_metrics=True):
        self.name = "ra_discriminator_loss"
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.track_metrics = track_metrics
        self.weighted = weighted

        self.loss_weight = loss_weight
        if not self.weighted:
            self.loss_weight = 1.0

        self.real_sample_loss = 0
        self.fake_sample_loss = 0
        self.total_loss = 0
        self.weighted_loss = 0
        log.debug(f"initialized relativistic discriminator loss - weighted: {self.weighted}, loss weight: {self.loss_weight}")

    def __call__(self, sr_critic, hr_critic, sr_labels, hr_labels, batch_metrics,
                 epoch_metrics):
        """
        Calculate relativistic average Discriminator loss based on real data samples and synthesized samples.

        :param sr_critic: Discriminators critique of synthesized High-Resolution samples from Generator.
        :param hr_critic: Discriminators critique of corresponding real data High-Resolution samples.
        :param sr_labels: Labels for synthesized samples to compare to Discriminators critique.
        :param hr_labels: Labels for real data samples to compare to Discriminators critique.
        :param batch_metrics: Optional dictionary to store batch metrics.
        :param epoch_metrics: Optional dictionary to store epoch metrics.
        :return: (Weighted) relativistic average Discriminator loss for batch.
        """
        self.real_sample_loss = self._cross_entropy(
            hr_labels,
            hr_critic - tf.reduce_mean(sr_critic)
        )

        self.fake_sample_loss = self._cross_entropy(
            sr_labels,
            sr_critic - tf.reduce_mean(hr_critic)
        )

        self.total_loss = (self.real_sample_loss + self.fake_sample_loss) / 2
        self.weighted_loss = self.total_loss * self.loss_weight

        if self.track_metrics:
            batch_metrics[self.name](self.total_loss)
            epoch_metrics[self.name](self.total_loss)
            if self.weighted:
                batch_metrics[f"weighted_{self.name}"](self.weighted_loss)
                epoch_metrics[f"weighted_{self.name}"](self.weighted_loss)
        return self.weighted_loss

    def __str__(self):
        return f"## Relativistic Discriminator Loss\n" \
               f"weighted: {self.weighted}\n" \
               f"loss weight: {self.loss_weight}\n"

