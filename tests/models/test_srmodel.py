import unittest
from unittest.mock import MagicMock
import tensorflow as tf
from simple_sr.models.sr_model import SRModel
from simple_sr.models.generator import Generator
from simple_sr.utils.models.loss_functions.mean_squared_error import MeanSquaredError
from simple_sr.utils.image import metrics
from tests import _utils

INITIAL_STEP = 0
INITIAL_METRIC_VAL = -1
FIRST_CHECKPOINT_STEP_DELTA = 5
FIRST_CHECKPOINT_METRIC_VAL = 25
SECOND_CHECKPOINT_STEP_DELTA = 1
SECOND_CHECKPOINT_METRIC_VAL = 28


class TestSRModel(unittest.TestCase):

    def test_restore_checkpoint(self):
        # build new SRResnet
        loss_func = MeanSquaredError()
        optimizer = tf.keras.optimizers.RMSprop
        self.model = self._load_model(
            loss_func=loss_func, optimizer=optimizer
        )
        self._assert_checkpoint(
            self.model, INITIAL_STEP, INITIAL_METRIC_VAL, generator=self.model.generator(),
            generator_optimizer=None,
        )
        self.assertEqual(type(self.model.generator_optimizer()), optimizer)

        # save checkpoint
        self.model._checkpoint_manager.save()

        # change checkpoint
        self._change_checkpoint(
            self.model, FIRST_CHECKPOINT_STEP_DELTA, FIRST_CHECKPOINT_METRIC_VAL
        )
        self._assert_checkpoint(
            self.model, FIRST_CHECKPOINT_STEP_DELTA, FIRST_CHECKPOINT_METRIC_VAL,
            generator=self.model.generator()
        )
        self.assertEqual(type(self.model.generator_optimizer()), optimizer)

        # restore first checkpoint
        self.model._checkpoint.restore(self.model._checkpoint_manager.latest_checkpoint)
        self._assert_checkpoint(
            self.model, INITIAL_STEP, INITIAL_METRIC_VAL, generator=self.model.generator()
        )
        self.assertEqual(type(self.model.generator_optimizer()), optimizer)

    def test_load_model_from_checkpoint(self):
        first_model_loss_func = MeanSquaredError()
        first_model_optimizer = tf.keras.optimizers.RMSprop
        first_model = self._load_model(
            loss_func=first_model_loss_func, optimizer=first_model_optimizer
        )
        self._assert_checkpoint(
            first_model, INITIAL_STEP, INITIAL_METRIC_VAL, generator=first_model.generator()
        )
        self.assertEqual(
            type(first_model._generator.loss_functions()[0]),
            type(MeanSquaredError())
        )
        self.assertEqual(type(first_model.generator_optimizer()), first_model_optimizer)

        self._change_checkpoint(
            first_model, FIRST_CHECKPOINT_STEP_DELTA, FIRST_CHECKPOINT_METRIC_VAL
        )
        self._assert_checkpoint(
            first_model, FIRST_CHECKPOINT_STEP_DELTA, FIRST_CHECKPOINT_METRIC_VAL,
            generator=first_model.generator()
        )
        self.assertEqual(type(first_model.generator_optimizer()), first_model_optimizer)
        first_model._checkpoint_manager.save()

        # create new model from latest checkpoint
        new_model = self._load_model(
            loss_func=MeanSquaredError(), optimizer=None,
            checkpoint=first_model.latest_checkpoint()
        )
        self._assert_checkpoint(
            new_model, FIRST_CHECKPOINT_STEP_DELTA, FIRST_CHECKPOINT_METRIC_VAL,
            generator=first_model.generator(),
            generator_optimizer=first_model.generator_optimizer()
        )

        # change checkpoint again
        self._change_checkpoint(
            new_model, SECOND_CHECKPOINT_STEP_DELTA, SECOND_CHECKPOINT_METRIC_VAL
        )
        self._assert_checkpoint(
            new_model, FIRST_CHECKPOINT_STEP_DELTA + SECOND_CHECKPOINT_STEP_DELTA,
            SECOND_CHECKPOINT_METRIC_VAL,
            generator=first_model.generator(),
            generator_optimizer=first_model.generator_optimizer()
        )

    def test_metrics(self):
        # load images to calc metrics on
        batch1, batch2 = _utils.load_img_batches()

        # load model without specifiying metrics
        m = self._load_model(loss_func=MeanSquaredError(), optimizer=tf.keras.optimizers.Adam)
        expected_metrics = dict(psnr=metrics.psnr)
        self._assert_metrics_initialized(m, expected_metrics)

        # update metrics via model and manually calc psnr for comparison
        for b1, b2 in zip(batch1.take(1), batch2.take(1)):
            m._update_metrics(b1, b2, m._train_epoch_metrics)
            m._update_metrics(b1, b2, m._valid_epoch_metrics)
            _psnr = tf.reduce_mean(metrics.psnr(b1, b2, max_val=2.0))
        expected_values = dict(psnr=_psnr)
        self._assert_metric_values(m._train_epoch_metrics, m._batch_metrics, expected_values, expected_values)
        self._assert_metric_values(m._valid_epoch_metrics, m._batch_metrics, expected_values, expected_values)

        self.assertEqual(0, len(m._train_batch_history["psnr"]))
        # call after_train_batch -> should update train batch history and reset batch metrics
        m.after_train_batch()
        self.assertEqual(1, len(m._train_batch_history["psnr"]))
        self.assertEqual(_psnr, m._train_batch_history["psnr"])
        self._assert_metric_values(m._train_epoch_metrics, m._batch_metrics,
                                   expected_epoch_values=expected_values,
                                   expected_batch_values=dict(psnr=0.0))
        self._assert_metric_values(m._valid_epoch_metrics, m._batch_metrics,
                                   expected_epoch_values=expected_values,
                                   expected_batch_values=dict(psnr=0.0))

        # test with all metrics from image.metrics module
        expected_metrics = dict(psnr=metrics.psnr, PSNR_Y=metrics.psnr_on_y, SSIM=metrics.ssim)
        m2 = self._load_model(loss_func=MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam, metrics=expected_metrics)
        self._assert_metrics_initialized(m2, expected_metrics)

        for b1, b2 in zip(batch1.take(1), batch2.take(1)):
            m2._update_metrics(b1, b2, m2._train_epoch_metrics)
            m2._update_metrics(b1, b2, m2._valid_epoch_metrics)
            _psnr = tf.reduce_mean(metrics.psnr(b1, b2, max_val=2.0))
            _psnr_y = tf.reduce_mean(metrics.psnr_on_y(b1, b2, max_val=2.0))
            _ssim = tf.reduce_mean(metrics.ssim(b1, b2, max_val=2.0))
        expected_values = dict(
            psnr=_psnr, PSNR_Y=_psnr_y, SSIM=_ssim
        )
        self._assert_metric_values(m2._train_epoch_metrics, m2._batch_metrics, expected_values, expected_values)
        self._assert_metric_values(m2._valid_epoch_metrics, m2._batch_metrics, expected_values, expected_values)

        # test with custom metric
        expected_metrics = dict(PSNR_Y=metrics.psnr_on_y, ADD=lambda img1, img2: img1 + img2)
        m3 = self._load_model(loss_func=MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam, metrics=expected_metrics)
        self._assert_metrics_initialized(m3, expected_metrics)

        for b1, b2 in zip(batch1.take(1), batch2.take(1)):
            m3._update_metrics(b1, b2, m3._train_epoch_metrics)
            _psnr_y = tf.reduce_mean(metrics.psnr_on_y(b1, b2, max_val=2.0))
            _add = tf.reduce_mean(b1 + b2)
        expected_values = dict(PSNR_Y=_psnr_y, ADD=_add)
        self._assert_metric_values(m3._train_epoch_metrics, m3._batch_metrics, expected_values, expected_values)
        self._assert_metric_values(m3._valid_epoch_metrics, m3._batch_metrics, dict(PSNR_Y=0.0, ADD=0.0),
                                   expected_values)

        # test with non-normalized imgs
        batch1, batch2 = _utils.load_img_batches(normalize=False)
        expected_metrics = dict(PSNR=lambda img1, img2: metrics.psnr(img1, img2, max_val=255))
        m4 = self._load_model(loss_func=MeanSquaredError(),
                              optimizer=tf.keras.optimizers.Adam, metrics=expected_metrics)
        self._assert_metrics_initialized(m4, expected_metrics)

        for b1, b2 in zip(batch1.take(1), batch2.take(1)):
            m4._update_metrics(b1, b2, m4._train_epoch_metrics)
            _psnr = tf.reduce_mean(metrics.psnr(b1, b2, max_val=255))
        expected_values = dict(PSNR=_psnr)
        self._assert_metric_values(m4._train_epoch_metrics, m4._batch_metrics, expected_values, expected_values)

    def _assert_metric_values(self, model_epoch_metrics, model_batch_metrics,
                              expected_epoch_values, expected_batch_values):
        for key, val in expected_epoch_values.items():
            self.assertEqual(val, model_epoch_metrics[key].result())
        for key, val in expected_batch_values.items():
            self.assertEqual(val, model_batch_metrics[key].result())

    def _assert_metrics_initialized(self, model, expected_metrics):
        self.assertEqual(len(expected_metrics), len(model._image_metrics))
        for key, func in expected_metrics.items():
            self.assertTrue(key in model._image_metrics.keys())
            self.assertEqual(func, model._image_metrics[key])
            self.assertTrue(key in model._train_epoch_metrics.keys())
            self.assertEqual(type(model._train_epoch_metrics[key]), tf.keras.metrics.Mean)
            self.assertTrue(key in model._valid_epoch_metrics.keys())
            self.assertEqual(type(model._valid_epoch_metrics[key]), tf.keras.metrics.Mean)
            self.assertTrue(key in model._batch_metrics.keys())
            self.assertEqual(type(model._batch_metrics[key]), tf.keras.metrics.Mean)

    def _assert_checkpoint(self, srmodel, step, metric_val, generator,
                           generator_optimizer=None, discriminator=None):
        self.assertEqual(srmodel.latest_checkpoint().step, step)
        self.assertEqual(srmodel.latest_checkpoint().metric.numpy(), metric_val)
        if generator_optimizer is not None:
            self.assertEqual(srmodel.latest_checkpoint().generator_optimizer, generator_optimizer)
        self.assertEqual(srmodel.latest_checkpoint().generator, generator)
        if discriminator is not None:
            self.assertEqual(srmodel.latest_checkpoint().discriminator, discriminator)

    def _change_checkpoint(self, srmodel, steps_to_add, metric_val):
        srmodel.latest_checkpoint().step.assign_add(steps_to_add)
        srmodel.latest_checkpoint().metric = tf.Variable(metric_val)

    def _load_model(self, loss_func, optimizer, checkpoint=None, loaded_model=None, metrics=None):

        model = SRModel(
            "resnet",
            Generator(
                upsample_factor=2, architecture="srresnet", loss_functions=loss_func
            ),
            generator_optimizer=optimizer,
            image_metrics=metrics,
            resnet_checkpoint=checkpoint
        )
        model._log_epoch_metrics_to_TB = MagicMock(return_value=0)
        model._log_batch_metrics_to_TB = MagicMock(return_value=0)
        return model


if __name__ == "__main__":
    unittest.main()
