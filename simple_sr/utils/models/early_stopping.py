

class EarlyStopping:
    def __init__(self, metric_key, patience):
        self._metric_key = metric_key
        self._patience = patience
        self._epochs_without_improvement = 0
        self._num_epochs_after_best = 0
        self._early_stop = False
        self._current_best_val = float("-inf")

    def evaluate_stop_criterion(self, metric_history):
        metric_this_epoch = metric_history[-1]
        try:
            metric_last_epoch = metric_history[-2]
        except IndexError:
            metric_last_epoch = float("-inf")
        if metric_this_epoch > self._current_best_val:
            self._epochs_without_improvement = 0
            self._num_epochs_after_best = 0
            self._current_best_val = metric_this_epoch
        else:
            self._num_epochs_after_best += 1
            if metric_this_epoch < metric_last_epoch:
                self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self._patience:
            self._early_stop = True

    def stop_early(self):
        return self._early_stop

    def num_epochs_after_best(self):
        return self._num_epochs_after_best

    def epochs_without_improvement(self):
        return self._epochs_without_improvement
