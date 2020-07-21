import unittest
import tensorflow as tf
from simple_sr.models.discriminator import Discriminator
from simple_sr.utils.models.loss_functions.discriminator_loss import DiscriminatorLoss


class TestDiscriminator(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_label_smoothing(self):
        smoothing_offset = 0.3
        disc = Discriminator(
            loss_function=DiscriminatorLoss(weighted=False),
            relativistic=False,
            label_smoothing=True,
            smoothing_offset=smoothing_offset,
            input_dims=(80, 80)
        )

        sr_critic = tf.ones(shape=(50,), dtype=tf.float64)
        hr_critic = tf.zeros(shape=(50,), dtype=tf.float64)

        sr_labels, hr_labels = disc._get_labels(sr_critic, hr_critic)

        self.assertEqual(len(sr_critic), len(sr_labels))
        self.assertEqual(len(hr_critic), len(hr_labels))

        self.assertGreaterEqual(tf.reduce_min(sr_labels).numpy(), 0)
        self.assertLessEqual(tf.reduce_max(sr_labels).numpy(), smoothing_offset)
        self.assertGreater(tf.math.reduce_std(sr_labels).numpy(), 0)

        self.assertGreaterEqual(tf.reduce_min(hr_labels).numpy(), (1 - smoothing_offset))
        self.assertLessEqual(tf.reduce_max(hr_labels).numpy(), (1 + smoothing_offset))
        self.assertGreater(tf.math.reduce_std(hr_labels).numpy(), 0)

    def test_non_smooth_labels(self):
        disc = Discriminator(
            loss_function=DiscriminatorLoss(weighted=False),
            relativistic=False,
            label_smoothing=False,
            input_dims=(80, 80)
        )

        sr_critic = tf.ones(shape=(50,), dtype=tf.float64)
        hr_critic = tf.zeros(shape=(50,), dtype=tf.float64)

        sr_labels, hr_labels = disc._get_labels(sr_critic, hr_critic)

        self.assertEqual(len(sr_critic), len(sr_labels))
        self.assertEqual(len(hr_critic), len(hr_labels))

        self.assertEqual(0, tf.reduce_min(sr_labels).numpy())
        self.assertEqual(0, tf.reduce_max(sr_labels).numpy())
        self.assertEqual(0, tf.reduce_mean(sr_labels).numpy())
        self.assertEqual(0, tf.math.reduce_std(sr_labels).numpy())

        self.assertEqual(1, tf.reduce_min(hr_labels).numpy())
        self.assertEqual(1, tf.reduce_max(hr_labels).numpy())
        self.assertEqual(1, tf.reduce_mean(hr_labels).numpy())
        self.assertEqual(0, tf.math.reduce_std(hr_labels).numpy())


if __name__ == "__main__":
    unittest.main()
