import os
from tqdm import tqdm
import time
import logging
import tensorflow as tf
from simple_sr.operations import evaluation
from simple_sr.utils import logger
from simple_sr.utils.train_result import TrainResult

log = logging.getLogger(logger.RESULTS_LOGGER)


def run_training(config, data_pipeline, sr_model, plotting_interval=1):
    """
    Run training session with initialized `DataPipeline` and `SRModel`.

    :param config: Initialized :code:`SimpleSR.ConfigUtil` object.
    :param data_pipeline: Initialized :code:`SimpleSR.DataPipeline` object.
    :param sr_model: Initialized :code:`SimpleSR.SRModel` object.
    :param plotting_interval:
        | Interval of epochs to plot and save images.
        | Every batch of the test set will be plotted (interpolated and
          upsampled by model respectively) and saved. Additionally one batch of
          training and validation set will be plotted as well.

        .. note::

            Plotting can be helpful to monitor progress during training,
            but will impact performance and therefore lead
            to longer epoch times.
    """
    _log_configurations(config, data_pipeline, sr_model)
    start_training = time.perf_counter()

    for epoch in tqdm(range(config.num_epochs)):
        learn_rate_info = \
            f"optimizer step: {sr_model.generator_optimizer().iterations.numpy()}, "\
            f"current learnrate: {sr_model.generator_optimizer()._decayed_lr(tf.float32).numpy()}"
        log.info(learn_rate_info)
        if sr_model.stop_early():
            break
        sr_model.before_epoch()
        start_epoch = time.perf_counter()

        # training
        train_batch_generator = data_pipeline.train_batch_generator()
        for step, (lr_batch, hr_batch) in enumerate(train_batch_generator):
            sr_model.train_step(lr_batch, hr_batch)
            sr_model.after_train_batch()

        # validation
        validation_batch_generator = data_pipeline.validation_batch_generator()
        for step, (lr_batch, hr_batch) in enumerate(validation_batch_generator):
            sr_model.validation_step(lr_batch, hr_batch)
            sr_model.after_validation_batch()

        if epoch != 0 and epoch % plotting_interval == 0:
            # test and plot on one train set batch
            train_batch_generator = data_pipeline.train_batch_generator()
            for _lr_batch, _hr_batch in train_batch_generator.take(1):
                lr_batch = _lr_batch
                hr_batch = _hr_batch
            sr_model.test_and_plot(
                lr_batch, config.pic_dir_train, 0, hr_batch,
            )

            # test and plot on one validation set batch
            validation_batch_generator = data_pipeline.validation_batch_generator()
            if type(validation_batch_generator) != list:
                for _lr_batch, _hr_batch in validation_batch_generator.take(1):
                    lr_batch = _lr_batch
                    hr_batch = _hr_batch
                sr_model.test_and_plot(
                    lr_batch, config.pic_dir_val, 0, hr_batch,
                )

            # test and plot on testing data
            try:
                test_batch_generator = data_pipeline.test_batch_generator(batch_size=config.batch_size)
                for step, (lr_batch, file_path) in enumerate(test_batch_generator):
                    sr_model.test_and_plot(
                        lr_batch, config.pic_dir_test, step, None,
                    )
            except tf.errors.InvalidArgumentError:
                test_batch_generator = data_pipeline.test_batch_generator(batch_size=1)
                for step, (lr_batch, file_path) in enumerate(test_batch_generator):
                    sr_model.test_and_plot(
                        lr_batch, config.pic_dir_test, step, None,
                    )

        epoch_duration = time.perf_counter() - start_epoch
        log_string = _prepare_log_string(sr_model, epoch, epoch_duration)
        log.info(log_string)
        sr_model.after_epoch()

    training_duration = time.perf_counter() - start_training
    log.info(f"finished training ({training_duration:.2f} sec)")
    sr_model.after_training()
    exp_res = TrainResult(
        sr_model.epoch_history(train=True),
        sr_model.epoch_history(train=False),
        sr_model.batch_history(train=True),
        sr_model.batch_history(train=False)
    )
    os.makedirs(os.path.join(config.save_path, "json_dump"), exist_ok=True)
    exp_res.save_as_json(os.path.join(config.save_path, "json_dump"))

    log.info("calculating psnr/ssim on validation set")
    evaluation.evaluate_on_validationdata(
        config=config, model_name=sr_model.name, pipeline=data_pipeline,
        model=sr_model.generator()
    )
    log.info("done")


def _log_configurations(config, data_pipeline, model_config):
    with open(config.config_logfile, "a+") as f:
        f.write("# Base config\n")
        f.write(f"{config}\n")

        f.write("# Data Pipeline config\n")
        f.write(f"{data_pipeline}\n")

        f.write("# Model config\n")
        f.write(f"{model_config}\n")


def _prepare_log_string(sr_model, epoch, epoch_duration):
    log_string = f"epoch: {epoch} ({epoch_duration:.2f} sec)\n"
    log_string += sr_model.formatted_epoch_metrics()
    return log_string

