import tensorflow as tf
import logging
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class RaAdversarialLoss:
    """
    | Relativistic average Adversarial loss function for Generator in relativistic GAN setting.
    | After initialization the `RaAdversarialLoss` object can be used as
      a functor to calculate adversarial loss of the Generator:

    .. code::

        ra_loss = RaAdversarialLoss()
        ...
        loss = ra_loss(hr_batch, sr_batch, hr_critic, sr_critic)

    | If `track_metrics` is True, supplied metrics dictionaries will updated with calculated loss.

    :param weighted: whether loss should be weighted
    :param loss_weight: weight factor for loss
    :param track_metrics: whether the class should update the supplied metrics dictionaries.
    """
    def __init__(self, weighted=False, loss_weight=1.0, track_metrics=True):
        self.name = "ra_adversarial_loss"
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
        log.debug(f"initialized relativistic adversarial loss - weighted: {self.weighted}, loss weight: {self.loss_weight}")

    def __call__(self, hr_batch, sr_batch, hr_critic, sr_critic, batch_metrics, epoch_metrics):
        """
        Calculate relativistic average adversarial loss for Generator from discriminators critique.

        .. note::
            The parameters `hr_batch` and `sr_batch` will not be used/needed for calculation
            of relativistic average adversarial loss, but the function needs to adhere to the (implicit) Generator
            loss function interface.

        :param hr_batch: Not needed, may be `None`
        :param sr_batch:  Not needed, may be `None`
        :param hr_critic: Discriminators critique of real data High-Resolution samples.
        :param sr_critic: Discriminators critique of generated High-Resolution samples.
        :param batch_metrics: Optional dictionary to store batch metrics.
        :param epoch_metrics: Optional dictionary to store epoch metrics.
        :return: (Weighted) relativistic average adversarial loss for batch.
        """
        self.real_sample_loss = self._cross_entropy(
            tf.zeros_like(hr_critic),
            hr_critic - tf.reduce_mean(sr_critic)
        )

        self.fake_sample_loss = self._cross_entropy(
            tf.ones_like(sr_critic),
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
        return f"## Relativistic Adversarial Loss\n" \
               f"weighted: {self.weighted}\n" \
               f"loss weight: {self.loss_weight}\n"
