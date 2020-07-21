import tensorflow as tf
import logging
from pathlib import Path
from simple_sr.utils.models.early_stopping import EarlyStopping
from simple_sr.utils.image import metrics, image_utils, image_transforms
from simple_sr.utils import logger

log = logging.getLogger(logger.LIB_LOGGER).getChild(__name__)
result_logger = logging.getLogger(logger.RESULTS_LOGGER)


class SRModel:
    """
    | `SRModel` encapsulates a `Generator` and optionally a `Discriminator`.
    | If no `Discriminator` is supplied the training will be non-adversarial mode, otherwise
      in adversarial mode.
    | The `SRModel` class is the main interface to interact with the `Generator` and `Discriminator`
      during training.
    | It manages the training process by delegating data batches to `Generator`/`Discriminator`,
      calculates gradients and updates weights of `Generator/Discriminator` accordingly.
    | Additionally the `SRModel` logs metrics to tensorboard/stdout, saves checkpoints
      and keeps track whether early stopping criterion is reached.

    :param model_type:
        Whether to train in 'gan' (adversarial) mode or 'resnet' (non-adversarial) mode.
    :param generator: Initialized `Generator` object.
    :param generator_optimizer: Optimizer for `Generator`.
    :param generator_optimizer_config:
        | Optimizer config for `Generator`, can specify things like learn-rate, learn-rate schedule etc.
          The optimizer config needs to be applicable to the supplied optimizer.
        | See the tensorflow docs and examples/complex_example.yaml for more details on this.
        | If no optimizer config is supplied the optimizer will be initialized with default values.
    :param discriminator:
        Optional `Discriminator` to train in adversarial mode.
    :param discriminator_optimizer:
        Optimizer for `Discriminator`, needs to be supplied if discriminator is supplied.
    :param discriminator_optimizer_config:
        Optimizer config for `Discriminator`, same things apply as for `generator_optimizer_config`.
    :param image_metrics:
        Dictionary containing pairs of `name: f(img1, img2)`. `f` will be calculated after every processed batch
        and logged to tensorboard. Average will be logged to stdout after each epoch.
        Defaults to :code:`{"PSNR": simple_sr.utils.image.metrics.psnr}`.
    :param early_stop_metric:
        Metric to track as trigger for early stopping.
    :param early_stop_patience:
        Defines how many epochs may pass without increasing `early_stop_metric`.
    :param epoch_train_summary_writer:
        Tensorflow summary writer for epoch training metrics.
    :param batch_train_summary_writer:
        Tensorflow summary writer for batch training metrics.
    :param epoch_validation_summary_writer:
        Tensorflow summary writer for epoch validation metrics.
    :param batch_validation_summary_writer:
        Tensorflow summary writer for batch validation metrics.
    :param resnet_checkpoint:
        Checkpoint of pretrained resnet model, the model of the generator will be
        set to the restored model.

        .. note::
            The `Generator` still needs to be initialized and supplied to the SRModel beforehand
            since only the `Generators` (keras)-model will be restored. So there is still the need
            to have a `Generator` with initialized loss functions.
    :param config:
        will be used to define save dirs, if not supplied base save dir defaults to "./".
    """

    def __init__(self,
                 model_type,
                 generator,
                 generator_optimizer,
                 generator_optimizer_config=None,
                 discriminator=None,
                 discriminator_optimizer=None,
                 discriminator_optimizer_config=None,
                 image_metrics=None,
                 early_stop_metric="psnr",
                 early_stop_patience=100,
                 epoch_train_summary_writer=None,
                 batch_train_summary_writer=None,
                 epoch_validation_summary_writer=None,
                 batch_validation_summary_writer=None,
                 resnet_checkpoint=None,
                 config=None):
        if model_type.lower() not in ["gan", "resnet"]:
            raise ValueError("model type not recognized")
        if generator is None:
            raise ValueError("no generator was supplied")
        if generator_optimizer is None and resnet_checkpoint is None:
            raise ValueError("no generator optimizer was supplied")
        if model_type == "gan" and discriminator is None:
            raise ValueError("model type is GAN but no discriminator supplied")
        if model_type == "gan" and discriminator_optimizer is None:
            raise ValueError("model type is GAN but no discriminator optimizer supplied")
        if model_type == "resnet" and discriminator is not None:
            raise ValueError("model type is Resnet but discriminator was supplied")

        self._model_type = model_type.lower()
        log.debug(f"training in {self._model_type} mode")
        self.name = model_type
        self._early_stop_metric = early_stop_metric
        self._early_stop_patience = early_stop_patience
        self._epochs = 0
        self._iterations = 0
        self._epoch_train_summary_writer = epoch_train_summary_writer
        self._batch_train_summary_writer = batch_train_summary_writer
        self._epoch_validation_summary_writer = epoch_validation_summary_writer
        self._batch_validation_summary_writer = batch_validation_summary_writer

        self._model_dir = "./models"
        self._checkpoint_dir = "./checkpoints"
        self._config = config
        if self._config is not None:
            if self._config.model_dir is not None:
                self._model_dir = self._config.model_dir
            if self._config.checkpoint_dir is not None:
                self._checkpoint_dir = self._config.checkpoint_dir

        # set up generator and generator optimizer
        self._generator = generator
        self._generator_loss_functions = generator.loss_functions()
        if generator_optimizer is not None and generator_optimizer_config is not None:
            log.debug("found generator optimizer config")
            try:
                self._generator_optimizer = generator_optimizer.from_config(*generator_optimizer_config)
            except TypeError:
                log.debug("couldn't initialize generator optimizer from config - trying a different way")
                self._generator_optimizer = generator_optimizer.from_config(generator_optimizer_config)
            log.debug("intialized generator optimizer from config")
        elif generator_optimizer is not None:
            self._generator_optimizer = generator_optimizer()
            log.debug("no config found for generator optimizer - initialized without config")
        self._generator_optimizer_config = generator_optimizer_config

        # set up discriminator and discriminator optimizer
        self._discriminator = discriminator
        self._discriminator_loss_functions = None
        if self._model_type == "gan":
            self._discriminator_loss_functions = discriminator.loss_function()
            if discriminator_optimizer_config is not None:
                log.debug("found discriminator optimizer config")
                try:
                    self._discriminator_optimizer = discriminator_optimizer.from_config(
                        *discriminator_optimizer_config
                    )
                except TypeError:
                    log.debug("couldn't initialize discriminator optimizer from config - trying a different way")
                    self._discriminator_optimizer = discriminator_optimizer.from_config(
                        discriminator_optimizer_config
                    )
                log.debug("intialized discriminator optimizer from config")
            else:
                self._discriminator_optimizer = discriminator_optimizer()
                log.debug("no config found for discriminator optimizer - initialized without config")
        self._discriminator_optimizer_config = discriminator_optimizer_config

        _step = None
        _metric = None
        if resnet_checkpoint is not None:
            log.debug("found resnet checkpooint")
            tmp_chp_manager = tf.train.CheckpointManager(checkpoint=resnet_checkpoint, directory="./tmp",
                                                         max_to_keep=1)
            resnet_checkpoint.restore(tmp_chp_manager.latest_checkpoint)
            _step = resnet_checkpoint.step
            _metric = resnet_checkpoint.metric
            self._generator.set_model(resnet_checkpoint.generator)
            self._generator_optimizer = resnet_checkpoint.generator_optimizer
            log.debug("restored resnet checkpoint")

        # TODO: load Gan checkpoint

        # set checkpoints and checkpoint manager
        if self._model_type == "gan":
            self._checkpoint = tf.train.Checkpoint(
                step=tf.Variable(0) if _step is None else _step,
                metric=tf.Variable(-1) if _metric is None else _metric,
                generator=self._generator.model(),
                generator_optimizer=self._generator_optimizer,
                discriminator=self._discriminator.model(),
                discriminator_optimizer=self._discriminator_optimizer
            )
        else:
            self._checkpoint = tf.train.Checkpoint(
                step=tf.Variable(0) if _step is None else _step,
                metric=tf.Variable(-1) if _metric is None else _metric,
                generator=self._generator.model(),
                generator_optimizer=self._generator_optimizer,
            )

        self._checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self._checkpoint, directory=f"{self._checkpoint_dir}/{self._model_type}",
            max_to_keep=5
        )

        # initialize metrics
        self._train_epoch_metrics = dict()
        self._valid_epoch_metrics = dict()
        self._batch_metrics = dict()
        self._image_metrics = image_metrics if image_metrics is not None else dict(psnr=metrics.psnr)

        for key, func in self._image_metrics.items():
            self._train_epoch_metrics[key] = tf.keras.metrics.Mean()
            self._valid_epoch_metrics[key] = tf.keras.metrics.Mean()
            self._batch_metrics[key] = tf.keras.metrics.Mean()

        self._train_batch_history = self._init_metrics_history(self._combined_epoch_metrics(train=True))
        self._train_epoch_history = self._init_metrics_history(self._combined_epoch_metrics(train=True))
        self._valid_batch_history = self._init_metrics_history(self._combined_epoch_metrics(train=False))
        self._valid_epoch_history = self._init_metrics_history(self._combined_epoch_metrics(train=False))

        self._early_stop_metric = early_stop_metric
        self._early_stop_patience = early_stop_patience
        self._early_stopping_util = EarlyStopping(metric_key=self._early_stop_metric,
                                                  patience=self._early_stop_patience)

    def iterations(self):
        return self._iterations

    def latest_checkpoint(self):
        """
        Get the latest checkpoint that was saved.

        A checkpoint contains:
            * The current iteration step
            * The tracked early stop metric value
            * The `Generator` model
            * The `Discriminator` model (if training in GAN mode)

        :return:
            Tensorflow checkpoint object.
        """
        return self._checkpoint

    def save_model(self, save_path, postfix=None):
        """
        Saves the `Generator` model in hdf5 format to disk.

        :param save_path:
            Path to save `Generator` model to.
        :param postfix:
            Optional postfix for filename, if None current epoch will be prefixed.
        """
        if postfix is None:
            postfix = self._epochs
        self._generator.model().save(f"{save_path}/{self._model_type}_gen_{postfix}.h5")

    def stop_early(self):
        """
        Check whether early stopping criterion is reached.

        :return:
            True if early stopping is reached, otherwise False.
        """
        return self._early_stopping_util.stop_early()

    def generator(self):
        """
        Retrieve the current `Generator` model.

        Note: This only returns the Keras model of the `Generator` not
        the `Generator` object itself.

        :return:
            tf.keras.model instance of current `Generator` model.
        """
        return self._generator.model()

    def generator_optimizer(self):
        """
        Retrieve the initialized and configured generator optimizer.

        :return:
            Tensorflow/Keras optimizer object of `Generator`.
        """
        return self._generator_optimizer

    def discriminator(self):
        """
        Retrieve the current `Discriminator`.

        Note: This only returns the Keras model of the `Discriminator` not
        the `Discriminator` object itself.

        :return:
            tf.keras.model instance of current `Discriminator` model or None if training in non-adversarial mode.
        """
        if self._discriminator is not None:
            return self._discriminator.model()
        return None

    def discriminator_optimizer(self):
        """
        Retrieve the initialized and configured discriminator optimizer.

        :return:
            Tensorflow/Keras optimizer object of `Discriminator` or None if training in non-adversarial mode.
        """
        return self._discriminator_optimizer

    def epoch_metrics(self, train=True):
        """
        Retrieve training or validation epoch metrics of current epoch.
        Metrics will be reset after each epoch.

        :param train:
            If `train` is true training metrics will be returned, otherwise validation metrics.
        :return:
            A dictionary containing the current epochs training or validation metrics.
        """
        if train:
            return self._train_epoch_metrics
        else:
            return self._valid_epoch_metrics

    def _combined_epoch_metrics(self, train=True):
        if self._model_type == "gan":
            combined_metrics = {
                **self._generator.epoch_metrics(train=train),
                **self._discriminator.epoch_metrics(train=train),
                **self.epoch_metrics(train=train)
            }
        else:
            combined_metrics = {
                **self._generator.epoch_metrics(train=train),
                **self.epoch_metrics(train=train)
            }
        return combined_metrics

    def batch_metrics(self):
        """
        Retrieve the current batch metrics.
        Since metrics will be reset after each batch there is no need to make a distinction between
        training and validation batch metrics.

        :return:
            A dictionary containing the current batch metrics
        """
        return self._batch_metrics

    def epoch_history(self, train=True):
        """
        Retrieve epoch history of collected metrics.
        :param train: Whether to retrieve training or validation epoch history.
        :return: List of collected metrics.
        """
        if train:
            return self._train_epoch_history
        return self._valid_epoch_history

    def batch_history(self, train=True):
        """
        Retrieve batch history of collected metrics.
        :param train: Whether to retrieve training or validation batch history.
        :return: List of collected metrics.
        """
        if train:
            return self._train_batch_history
        return self._valid_batch_history

    def _combined_batch_metrics(self):
        if self._model_type == "gan":
            combined_metrics = {
                **self._generator.batch_metrics(),
                **self._discriminator.batch_metrics(),
                **self._batch_metrics
            }
        else:
            combined_metrics = {
                **self._generator.batch_metrics(),
                **self._batch_metrics
            }
        return combined_metrics

    def epoch_summary_writer(self, train=True):
        """
        Retrieve the training or validation epoch summary writer.

        :param train:
            If train is True the training epoch summary writer will be returned,
            otherwise the validation epoch summary writer.
        :return:
            Tensorflow summary writer object.
        """
        if train:
            return self._epoch_train_summary_writer
        else:
            return self._epoch_validation_summary_writer

    def batch_summary_writer(self, train=True):
        """
        Retrieve the training or validation batch summary writer.

        :param train:
            If train is True the training batch summary writer will be returned,
            otherwise the validation batch summary writer.
        :return:
            Tensorflow summary writer object.
        """
        if train:
            return self._batch_train_summary_writer
        else:
            return self._batch_validation_summary_writer

    @tf.function
    def train_step(self, lr_batch, hr_batch):
        """
        Train for one iteration on supplied batches.
        Generation of images and calculation of loss will be delegated to the `Generator`
        and afterwards the `SRModel` will calculate the `Generators` gradients and update
        the model of the `Generator` accordingly.

        If training in adversarial mode, the critique and calculation of its loss will be delegated to
        the `Discriminator` and weights of the discriminator model will be updated by `SRModel` afterwards.

        :param lr_batch:
            Batch of low resolution samples.
        :param hr_batch:
            Batch of corresponding high resolution ground truth samples.
        """
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            sr_batch = self._generator.generate(lr_batch)

            sr_critic = hr_critic = None
            if self._model_type == "gan":
                sr_critic, hr_critic = self._discriminator.critic_train_batch(
                    sr_batch, hr_batch
                )
                disc_loss = self._discriminator.calculate_train_loss(
                    sr_critic, hr_critic
                )

            gen_loss = self._generator.calculate_train_loss(
                sr_batch, hr_batch, sr_critic, hr_critic
            )

        # update generator
        generator_gradients = generator_tape.gradient(
            gen_loss, self._generator.model().trainable_variables
        )
        self._generator_optimizer.apply_gradients(
            zip(generator_gradients, self._generator.model().trainable_variables)
        )

        # update discriminator
        if self._model_type == "gan":
            discriminator_gradients = discriminator_tape.gradient(
                disc_loss, self._discriminator.model().trainable_variables
            )
            self._discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients,
                    self._discriminator.model().trainable_variables)
            )

        self._update_metrics(hr_batch, sr_batch, self._train_epoch_metrics)

    @tf.function
    def validation_step(self, lr_batch, hr_batch):
        """
        Validate for one iteration on supplied batches.
        Only loss and image metrics will be calculated, models will not be updated.

        :param lr_batch: batch of low resolution samples
        :param hr_batch: batch of corresponding high resolution ground truth samples
        """
        sr_batch = self._generator.generate(lr_batch, training=False)

        sr_critic = hr_critic = None
        disc_loss = None
        if self._model_type == "gan":
            sr_critic, hr_critic = self._discriminator.critic_validation_batch(
                sr_batch, hr_batch
            )
            disc_loss = self._discriminator.calculate_validation_loss(
                sr_critic, hr_critic
            )

        gen_loss = self._generator.calculate_validation_loss(
            sr_batch, hr_batch, sr_critic, hr_critic
        )

        self._update_metrics(hr_batch, sr_batch, self._valid_epoch_metrics)

    def test_and_plot(self, lr_batch, save_dir, step, hr_batch=None, file_path=None):
        """
        Generate high resolution samples with generator from a supplied low resolution batch
        and save resulting images as image grid to monitor progress during training.
        Additionally each sample will be upsampled with bicubic interpolation for comparision.

        This can either be done for batches from the test set where no ground truth is available
        or for batches from train/validation data with a corresponding ground truth.
        If the ground truth is supplied it will also be plotted on the image grid.

        :param lr_batch: batch of low resolution sampled
        :param save_dir: save dir for saving the resulting image grid
        :param step: epoch to manage/identify file names of saved image grids
        :param hr_batch: optional batch of high resolution ground truth samples
        :param file_path: optional save dir suffix for grouping image grids in specific folders
        """
        sr_batch = self._generator.generate(lr_batch, training=False)
        fname = f"{str(self._epochs).zfill(5)}_{str(step).zfill(3)}"
        kwargs = {
            self._model_type: sr_batch,
            "bicubic": image_transforms.resize(
                lr_batch, (sr_batch.shape[1], sr_batch.shape[2])
            )
        }
        if hr_batch is not None:
            kwargs["ground truth"] = hr_batch
        save_dir = f"{save_dir}/{self._model_type}"
        if file_path is not None:
            save_dir += f"/{Path(file_path.numpy()[0].decode('utf-8')).parent.stem}"
        image_utils.prepare_image_grid(
            save_dir=f"{save_dir}", fname=fname,
            original=None,
            psnr=None,
            low_res_key=None,
            **kwargs
        )

    def after_train_batch(self):
        """
        | Called after each training batch (if you're using SimpleSR training utils).
        | Updates number of iterations the model has trained for, logs training batch metrics to tensorboard
          and resets batch metrics afterwards.

        """
        self._iterations = self._generator_optimizer.iterations.numpy()
        self._log_batch_metrics_to_TB()
        self._update_history(self._combined_batch_metrics(), self._train_batch_history)
        self._reset_batch_metrics()

    def after_validation_batch(self):
        """
        | Called after each validation batch (if you're using SimpleSR training utils).
        | Logs validation batch metrics to tensorboard and resets batch metrics afterwards.
        """
        self._log_batch_metrics_to_TB(train=False)
        self._update_history(self._combined_batch_metrics(), self._valid_batch_history)
        self._reset_batch_metrics()

    def _log_batch_metrics_to_TB(self, train=True):
        summary_writer = self.batch_summary_writer(train=train)
        _metrics = self._combined_batch_metrics()
        with summary_writer.as_default():
            for name, metric in _metrics.items():
                tf.summary.scalar(f"{name}_batch", metric.result(), step=self._iterations)

    def _log_epoch_metrics_to_TB(self, train=True):
        summary_writer = self.epoch_summary_writer(train=train)
        _metrics = self._combined_epoch_metrics(train=train)
        with summary_writer.as_default():
            for name, metric in _metrics.items():
                tf.summary.scalar(name, metric.result(), step=self._epochs)

    def before_epoch(self):
        """
        | Called before each epoch (if you're using SimpleSR training utils).
        | Resets epoch metrics (training and validation) and increments number of trained epochs.
        """
        self._reset_epoch_metrics()
        self._epochs += 1
        self._checkpoint.step.assign_add(1)

    def after_epoch(self):
        """
        Called after each epoch (if you're using SimpleSR training utils).

        * saves (generator) model
        * logs epoch metrics to tensorboard and updates epoch metrics history
        * evaluates whether early stopping criterion is triggerd
        """
        self.save_model(self._model_dir)
        self._log_epoch_metrics_to_TB(train=True)
        self._log_epoch_metrics_to_TB(train=False)
        self._update_epoch_history()

        # TODO check if valid data is present (i.e. if validation set was supplied)
        # TODO use train metrics if not present
        self._checkpoint.metric = tf.Variable(
            self._valid_epoch_metrics[self._early_stop_metric].result()
        )
        if self._check_early_stopping():
            log.debug("received STOP EARLY - restoring best checkpoint")
            log.debug(
                f"checkpoint before restoring - step:{self._checkpoint.step.numpy()}, "
                f"{self._early_stop_metric}: {self._checkpoint.metric.numpy()}"
            )
            self._checkpoint.restore(self._checkpoint_manager.latest_checkpoint)
            log.debug(
                f"restored best checkpoint - step:{self._checkpoint.step.numpy()}, "
                f"{self._early_stop_metric}: {self._checkpoint.metric.numpy()}"
            )

        if self._early_stopping_util.num_epochs_after_best() == 0:
            result_logger.info("recorded new highest value for tracked metric - saving checkpoint")
            self._checkpoint_manager.save()
            result_logger.info(
                f"saved checkpoint - step: {self._checkpoint.step.numpy()}, "
                f"{self._early_stop_metric}: {self._checkpoint.metric.numpy()}"
            )

    def after_training(self):
        """
        | Called after training finishes (if you're using SimpleSR training utils).
        | Restores the best model and saves it with "best" postfix for identification.
        | Plots metrics afterwards.
        """
        self._checkpoint.restore(self._checkpoint_manager.latest_checkpoint)
        self.save_model(self._model_dir, postfix="best")
        self._reset_epoch_metrics()

    def formatted_epoch_metrics(self):
        """ Retrieve formatted epoch metrics/losses for logging """
        return self._format_metrics(self._train_epoch_metrics, "Training") \
               + self._format_metrics(self._valid_epoch_metrics, "Validation")

    def _check_early_stopping(self):
        self._early_stopping_util.evaluate_stop_criterion(
            self._valid_epoch_history[self._early_stop_metric]
        )
        if self.stop_early():
            return True
        return False

    def _init_metrics_history(self, metrics):
        history = dict()
        for name in metrics.keys():
            history[name] = list()
        return history

    def _update_metrics(self, hr_batch, sr_batch, epoch_metrics):
        for key, func in self._image_metrics.items():
            res = func(hr_batch, sr_batch)
            epoch_metrics[key](res)
            self._batch_metrics[key](res)

    def _update_epoch_history(self):
        self._update_history(self._combined_epoch_metrics(train=True), self._train_epoch_history)
        self._update_history(self._combined_epoch_metrics(train=False), self._valid_epoch_history)

    def _update_history(self, metrics, history, reset=False):
        for name, metric in metrics.items():
            history[name].append(metric.result().numpy())
            if reset:
                metric.reset_states()

    def _reset_epoch_metrics(self):
        self._reset_metrics(self._train_epoch_metrics)
        self._reset_metrics(self._valid_epoch_metrics)
        self._generator.reset_epoch_metrics()
        if self._model_type == "gan":
            self._discriminator.reset_epoch_metrics()

    def _reset_batch_metrics(self):
        self._reset_metrics(self._batch_metrics)
        self._generator.reset_batch_metrics()
        if self._model_type == "gan":
            self._discriminator.reset_batch_metrics()

    def _reset_metrics(self, metrics):
        for name, metric in metrics.items():
            metric.reset_states()

    def _format_metrics(self, metrics, header):
        img_metrics_info = ""
        for key, val in self._image_metrics.items():
            img_metrics_info += f"{key}: {metrics[key].result():.5f}\n"
        if header == "Training":
            gen_loss_info = self._generator.formatted_epoch_metrics(train=True)
        else:
            gen_loss_info = self._generator.formatted_epoch_metrics(train=False)
        if self._model_type == "gan":
            if header == "Training":
                disc_loss_info = self._discriminator.formatted_epoch_metrics(train=True)
            else:
                disc_loss_info = self._discriminator.formatted_epoch_metrics(train=False)
            return f"{header}\n" \
                   f"{img_metrics_info}"\
                   f"Generator\n" \
                   f"{gen_loss_info}" \
                   f"Discriminator\n" \
                   f"{disc_loss_info}"
        else:
            log_string = f"{header}\n "
            for name, metric in metrics.items():
                log_string += f"{name}: {metric.result():.4f}, "
            return log_string + "\n" + gen_loss_info + "\n"

    def __str__(self):
        disc_optimizer_config = None
        if self._discriminator is not None:
            disc_optimizer_config = self._discriminator_optimizer.get_config()
        return f"# SR Model\n"\
               f"model type: {self._model_type}\n"\
               f"generator optimizer: {self._generator_optimizer.get_config()}\n"\
               f"supplied generator optimizer config: {self._generator_optimizer_config}\n"\
               f"discriminator optimizer: {disc_optimizer_config}\n"\
               f"supplied discriminator optimizer config: {self._discriminator_optimizer_config}\n"\
               f"image metrics: {self._image_metrics}\n" \
               f"early stop metric: {self._early_stop_metric}\n"\
               f"early stop patience: {self._early_stop_patience}\n\n"\
               f"{self._generator}\n"\
               f"{self._discriminator}"

    @staticmethod
    def init(config,
             generator,
             generator_optimizer,
             generator_optimizer_config=None,
             discriminator=None,
             discriminator_optimizer=None,
             discriminator_optimizer_config=None,
             image_metrics=None):
        """
        Convenience method to initialize SRModel - model type will be inferred and early stopping as well
        as Tensorflow summary writers will used from initialized config.

        :return: Initialized Instance of type `SRModel`, ready for training.
        """
        if discriminator is None:
            model_type = "resnet"
        else:
            model_type = "gan"
        return SRModel(
            model_type=model_type,
            generator=generator,
            generator_optimizer=generator_optimizer,
            generator_optimizer_config=generator_optimizer_config,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            discriminator_optimizer_config=discriminator_optimizer_config,
            image_metrics=image_metrics,
            early_stop_metric=config.early_stop_metric,
            early_stop_patience=config.early_stop_patience,
            epoch_train_summary_writer=config.epoch_train_summary_writer,
            batch_train_summary_writer=config.batch_train_summary_writer,
            epoch_validation_summary_writer=config.epoch_validation_summary_writer,
            batch_validation_summary_writer=config.batch_validation_summary_writer,
            config=config
        )

