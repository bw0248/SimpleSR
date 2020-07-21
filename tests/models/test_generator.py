import unittest
import tensorflow as tf
from simple_sr.models.generator import Generator
from simple_sr.utils.models.loss_functions.mean_squared_error import MeanSquaredError
from tests import _utils


class TestGenerator(unittest.TestCase):

    def test_custom_loss_function_as_lambda(self):
        mse = MeanSquaredError(track_metrics=False)
        generator = Generator(
            upsample_factor=2, architecture="srresnet",
            loss_functions=[
                mse,
                lambda hr_batch, sr_batch, x, xx, xxx, xxxx: tf.reduce_max(hr_batch) + tf.reduce_min(sr_batch)
            ]
        )

        batch1, batch2 = _utils.load_img_batches()
        for b1, b2 in zip(batch1.take(1), batch2.take(2)):
            _mse = mse(b1, b2, None, None, None, None)
            _custom_metric = tf.reduce_max(b2) + tf.reduce_min(b1)
            _loss = generator.calculate_train_loss(b1, b2, None, None)
        self.assertEqual(_mse + _custom_metric, _loss)
        self.assertEqual(_mse + _custom_metric, generator.batch_metrics()["generator_loss"].result())
        self.assertEqual(_mse + _custom_metric, generator.epoch_metrics()["generator_loss"].result())

    def test_custom_loss_function_as_class(self):
        mse = MeanSquaredError(track_metrics=True)
        custom_loss_func = CustomLossFunctionTest(track_metrics=True)
        generator = Generator(
            upsample_factor=2, architecture="srresnet",
            loss_functions=[
                mse,
                custom_loss_func
            ]
        )

        batch1, batch2 = _utils.load_img_batches()
        for b1, b2 in zip(batch1.take(1), batch2.take(2)):
            _mse = tf.keras.metrics.MeanSquaredError()(b1, b2)
            _custom_metric = tf.reduce_max(4 * b2 + b1)
            _loss = generator.calculate_train_loss(b1, b2, None, None)
        self.assertEqual(_mse + _custom_metric, _loss)
        self.assertEqual(_mse + _custom_metric, generator.batch_metrics()["generator_loss"].result())
        self.assertEqual(_mse + _custom_metric, generator.epoch_metrics()["generator_loss"].result())
        self.assertEqual(_mse, generator.batch_metrics()[mse.name].result())
        self.assertEqual(_mse, generator.epoch_metrics()[mse.name].result())
        self.assertEqual(_custom_metric, generator.batch_metrics()[custom_loss_func.name].result())
        self.assertEqual(_custom_metric, generator.epoch_metrics()[custom_loss_func.name].result())


class CustomLossFunctionTest:
    def __init__(self, weighted=False, loss_weight=1.0, track_metrics=True):
        self.name = "custom_loss_func_test"
        self.track_metrics = track_metrics
        self.weighted = weighted
        self.loss_weight = loss_weight
        self.loss = 0
        self.weighted_loss = 0

    def __call__(self, hr_batch, sr_batch, hr_critic, sr_critic, batch_metrics, epoch_metrics):
        self.loss = tf.reduce_max(4 * hr_batch + sr_batch)
        self.weighted_loss = self.loss * self.loss_weight

        if self.track_metrics:
            batch_metrics[self.name](self.loss)
            epoch_metrics[self.name](self.loss)
            if self.weighted:
                batch_metrics[f"weighted_{self.name}"](self.weighted_loss)
                epoch_metrics[f"weighted_{self.name}"](self.weighted_loss)
        return self.weighted_loss


if __name__ == "__main__":
    unittest.main()






