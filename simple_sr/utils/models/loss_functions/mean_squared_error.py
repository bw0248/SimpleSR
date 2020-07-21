import tensorflow as tf
import logging
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)


class MeanSquaredError:
    """
    | Mean Squared Error function based on pixels.
    | After initialization the `MeanSquaredError` object can be used as
      a functor to calculate pixelwise mean squared error of generated images:

    .. code::

        mse = MeanSquaredError()
        ...
        loss = mse(hr_batch, sr_batch, hr_critic, sr_critic)

    | If `track_metrics` is True, supplied metrics dictionaries will updated with calculated loss.

    :param weighted: whether loss should be weighted
    :param loss_weight: weight factor for loss
    :param track_metrics: whether the class should update the supplied metrics dictionaries.
    """
    def __init__(self, weighted=False, loss_weight=1.0, track_metrics=True):
        self.name = "mean_squared_error"
        self.mse = tf.keras.losses.MeanSquaredError()
        self.track_metrics = track_metrics
        self.weighted = weighted

        self.loss_weight = loss_weight
        if not self.weighted:
            self.loss_weight = 1.0

        self.loss = 0
        self.weighted_loss = 0
        log.debug(f"initialized MSE - weighted: {self.weighted}, loss weight: {self.loss_weight}")

    def __call__(self, hr_batch, sr_batch, hr_critic, sr_critic, batch_metrics, epoch_metrics):
        """
        Calculate pixelwise mean squared error for supplied batches of images.

        .. note::
            The parameters `hr_critique` and `sr_critique` will not be used/needed for calculation
            of mean squared error, but the function needs to adhere to the (implicit) Generator
            loss function interface.

        :param hr_batch: Tensor of real data High-Resolution samples.
        :param sr_batch: Tensor of synthesized High-Resolution samples with equal shape as `hr_batch`.
        :param hr_critic: Not needed, may be `None`.
        :param sr_critic: Not needed, may be `None`.
        :param batch_metrics: Optional dictionary to store batch metrics.
        :param epoch_metrics: Optional dictionary to store epoch metrics.
        :return: (Weighted) mean squared error for batch.
        """
        self.loss = self.mse(hr_batch, sr_batch)
        self.weighted_loss = self.loss * self.loss_weight

        if self.track_metrics:
            batch_metrics[self.name](self.loss)
            epoch_metrics[self.name](self.loss)
            if self.weighted:
                batch_metrics[f"weighted_{self.name}"](self.weighted_loss)
                epoch_metrics[f"weighted_{self.name}"](self.weighted_loss)
        return self.weighted_loss

    def __str__(self):
        return f"## Mean Squared Error\n" \
               f"weighted: {self.weighted}\n" \
               f"loss weight: {self.loss_weight}\n"

