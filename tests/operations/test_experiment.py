import unittest

from simple_sr.utils.config.config_util import ConfigUtil
from simple_sr.operations.experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_something(self):
        base_conf = ConfigUtil.training_config(
            train_data_paths="./data/patterns/gradients",
            num_epochs=20,
            batch_size=8,
            scale=2,
            operation="experiment",
            crop_size=(64, 64, 3),
        )

        exp_param = [
            {
                "exp_1": {
                    "batch_size": 16,
                    "train_data_paths": "./data/patterns/random_noise"
                },
            },
            {
                "exp_2": {
                    "jpg_noise": True,
                    "jpg_noise_level": 80
                }
            },
            {
                "exp_3": {
                    "crop_size": (128, 128, 3)
                }
            }
        ]

        experiment = Experiment.initialize_experiment(
            base_conf, exp_param, include_base_config=True
        )

        num_configs = 0
        experiment_names = set()
        for config in experiment.next_config():
            num_configs += 1
            experiment_names.add(experiment.current_experiment_name)
            self.assertEqual(20, config.num_epochs)
            self.assertEqual(2, config.scale)
            self.assertEqual("experiment", config.operation)
            if experiment.current_experiment_name == "base config":
                self.assertEqual(8, config.batch_size)
                self.assertEqual(["./data/patterns/gradients"], config.train_data_paths)
                self.assertEqual((64, 64, 3), config.crop_size)
                self.assertFalse(config.jpg_noise)
            if experiment.current_experiment_name == "exp_1":
                self.assertEqual(16, config.batch_size)
                self.assertEqual("./data/patterns/random_noise", config.train_data_paths)
                self.assertEqual((64, 64, 3), config.crop_size)
                self.assertFalse(config.jpg_noise)
            if experiment.current_experiment_name == "exp_2":
                self.assertEqual(8, config.batch_size)
                self.assertEqual(["./data/patterns/gradients"], config.train_data_paths)
                self.assertEqual((64, 64, 3), config.crop_size)
                self.assertTrue(config.jpg_noise)
                self.assertEqual(80, config.jpg_noise_level)
            if experiment.current_experiment_name == "exp_3":
                self.assertEqual(8, config.batch_size)
                self.assertEqual(["./data/patterns/gradients"], config.train_data_paths)
                self.assertFalse(config.jpg_noise)
                self.assertEqual((128, 128, 3), config.crop_size)
        self.assertEqual(4, num_configs)
        self.assertEqual(4, len(experiment_names))


if __name__ == '__main__':
    unittest.main()
