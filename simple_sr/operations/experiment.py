import numpy as np
import json
import os
import copy

from simple_sr.utils.train_result import TrainResult
from simple_sr.utils.config.config_util import ConfigUtil


class Experiment:
    def __init__(self, base_config, experiment_params, include_base_config):
        self.base_config = base_config
        self.current_config = self.base_config
        self.current_experiment_name = None
        if include_base_config:
            self.experiment_params = [{"base config": {}}] + experiment_params
        else:
            self.experiment_params = experiment_params
        self.experiments_run = 0
        self.num_experiments = len(self.experiment_params)
        self.results = list()

    def next_config(self):
        while self.experiments_run < self.num_experiments:
            updated_config = copy.deepcopy(self.base_config)
            experiment = self.experiment_params[self.experiments_run]
            if len(experiment.keys()) != 1:
                raise ValueError("invalid experiment dict supplied")

            experiment_name = list(experiment.keys())[0]
            experiment_params = experiment[experiment_name]
            if len(experiment_params) > 0:
                updated_config.update_config(
                    **experiment_params
                )

            updated_config.save_path = f"{updated_config.save_path}/{experiment_name}"
            updated_config.reinitialize_save_dirs()

            self.current_config = updated_config
            self.current_experiment_name = experiment_name

            yield self.current_config
            self.experiments_run += 1

    def add_result(self, train_batch_history, valid_batch_history,
                   train_epoch_history, valid_epoch_history):
        self.results.append(
            TrainResult(
                train_epoch_history, valid_epoch_history,
                train_batch_history, valid_batch_history,
            )
        )

    def serialize_last_result(self, path):
        self.results[-1].save_as_json(path)

    @staticmethod
    def initialize_experiment(base_config, experiment_params, include_base_config=True):
        return Experiment(base_config, experiment_params, include_base_config)


if __name__ == "__main__":
    pass
