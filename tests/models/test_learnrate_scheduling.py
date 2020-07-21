import unittest
import tensorflow as tf
import numpy as np
from simple_sr.models.sr_model import SRModel
from simple_sr.models.generator import Generator
from simple_sr.data_pipeline.data_pipeline import DataPipeline
from simple_sr.utils.config.config_util import ConfigUtil

DATA_PATH = "./tests/data/patterns/random_noise"
BOUNDARIES = [2, 5]     # learn rate intervals
LEARN_RATES = [3e-4, 2e-5, 3e-6]    # learn rate values in interval
EXPECTED_LEARN_RATES_IN_EPOCH = [3e-4, 3e-4, 3e-4, 2e-5, 2e-5, 2e-5, 3e-6]
BETA_1 = 0.5
BETA_2 = 0.8
ALLOWED_DELTA = 1e-5
NUM_EPOCHS = 7


class TestLearnrateScheduling(unittest.TestCase):
    def test_learnrate(self):
        optimizer_config = {
            "learning_rate": {
                "class_name": "PiecewiseConstantDecay",
                "config": {
                    "boundaries": BOUNDARIES,
                    "values": LEARN_RATES
                },
            },
            "beta_1": 0.5,
            "beta_2": 0.8,
        }

        pipeline = DataPipeline(
            hr_img_path=DATA_PATH, validationset_path=DATA_PATH,
            batch_size=8, augmentations=[], scale=2,
            crop_size=(64, 64, 3)
        )

        model = SRModel(
            model_type="resnet",
            generator=Generator.srresnet(upsample_factor=2),
            generator_optimizer=tf.keras.optimizers.Adam,
            generator_optimizer_config=optimizer_config
        )

        self.assertEqual(BOUNDARIES, model.generator_optimizer()._get_hyper("learning_rate").boundaries)
        np.testing.assert_array_almost_equal(
            LEARN_RATES,
            model.generator_optimizer()._get_hyper("learning_rate").values,
            decimal=6
        )
        self.assertAlmostEqual(BETA_1, model.generator_optimizer()._get_hyper("beta_1").numpy(), delta=ALLOWED_DELTA)
        self.assertAlmostEqual(BETA_2, model.generator_optimizer()._get_hyper("beta_2").numpy(), delta=ALLOWED_DELTA)

        for i in range(NUM_EPOCHS):
            self._assert_optimizer(model.generator_optimizer(), EXPECTED_LEARN_RATES_IN_EPOCH[i], BETA_1, BETA_2)
            for lr, hr in pipeline.train_batch_generator().take(1):
                model.train_step(lr, hr)

    def _assert_optimizer(self, optimizer, expected_lr, expected_beta_1, expected_beta_2):
        self.assertAlmostEqual(expected_lr, optimizer._decayed_lr(tf.float32).numpy(), delta=ALLOWED_DELTA)
        self.assertAlmostEqual(expected_beta_1, optimizer._get_hyper("beta_1").numpy(), delta=ALLOWED_DELTA)
        self.assertAlmostEqual(expected_beta_2, optimizer._get_hyper("beta_2").numpy(), delta=ALLOWED_DELTA)


if __name__ == "__main__":
    unittest.main()
