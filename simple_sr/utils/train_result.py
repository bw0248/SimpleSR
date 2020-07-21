import os
import json


class TrainResult:
    """
    Helper class to serialize results of training sessions.

    :param train_epoch_history: Dictionary containing metric names as keys and a list of corresponding results as values.
    :param valid_epoch_history: Dictionary containing metric names as keys and a list of corresponding results as values.
    :param train_batch_history: Dictionary containing metric names as keys and a list of corresponding results as values.
    :param valid_batch_history: Dictionary containing metric names as keys and a list of corresponding results as values.
    """
    def __init__(self, train_epoch_history, valid_epoch_history,
                 train_batch_history, valid_batch_history):
        self.train_batch_history = train_batch_history
        self.valid_batch_history = valid_batch_history
        self.train_epoch_history = train_epoch_history
        self.valid_epoch_history = valid_epoch_history

    def save_as_json(self, path):
        """
        Write result to disk as json file.

        :param path: Path to save json file.
        """
        _train_epoch_history = self._convert_to_float(self.train_epoch_history)
        _val_epoch_history = self._convert_to_float(self.valid_epoch_history)
        _train_batch_history = self._convert_to_float(self.train_batch_history)
        _val_batch_history = self._convert_to_float(self.valid_batch_history)
        _res = TrainResult(
            _train_epoch_history, _val_epoch_history,
            _train_batch_history, _val_batch_history
        )
        for history_name in ["train_batch_history", "valid_batch_history",
                             "train_epoch_history", "valid_epoch_history"]:
            with open(os.path.join(path, history_name), "w") as f:
                json.dump(_res.__dict__[history_name], f)

    def _convert_to_float(self, history):
        sanitized = dict()
        for metric, result in history.items():
            sanitized[metric] = [float(x) for x in result]
        return sanitized
