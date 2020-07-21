import tensorflow as tf
from simple_sr.utils.config.config_util import ConfigUtil
from simple_sr.models.sr_model import SRModel
from simple_sr.models.generator import Generator
from simple_sr.models.discriminator import Discriminator
from simple_sr.utils.models.loss_functions.vgg_loss import VGGLoss
from simple_sr.utils.models.loss_functions.adversarial_loss import AdversarialLoss
from simple_sr.utils.models.loss_functions.mean_absolute_error import MeanAbsoluteError
from simple_sr.utils.models.loss_functions.ra_adversarial_loss import RaAdversarialLoss
from simple_sr.data_pipeline.data_pipeline import DataPipeline
from simple_sr.operations import training
from simple_sr.utils.image import image_transforms, metrics


# enter paths to data sets here
TRAIN_DATA = ""
VAL_DATA = ""
TEST_DATA = ""
SAVE_DIR = ""


"""
Examples for training configurations without using yaml files.
This might be preferable for consecutively pre-training Generators and then
using the pre-trained network for training in adversarial mode.
"""


def srresnet_model(pretrained_model=None):
    upsample_factor = 2
    config = ConfigUtil.training_config(
        train_data_paths=TRAIN_DATA,
        validation_data_path=VAL_DATA,
        results_save_path=SAVE_DIR,
        num_epochs=25,
        batch_size=8,
        resize_filter=tf.image.ResizeMethod.BICUBIC,
        scale=upsample_factor,
        test_data_path=TEST_DATA,
        train_val_split=0,
        crop_imgs=True,
        crop_size=(96, 96, 3),
        num_crops=20,
        augmentations=[
            image_transforms.rotate90, image_transforms.flip_along_x,
            image_transforms.flip_along_y
        ],
        jpg_noise=False,   
        jpg_noise_level=0,
        shuffle_buffer_size=4096,
        random_seed=None,
        early_stop_metric="PSNR",
        early_stop_patience=500
    )

    generator = Generator.srresnet(
        upsample_factor=upsample_factor,
        pretrained_model=pretrained_model
    )
    model = SRModel.init(
        config,
        generator=generator,
        generator_optimizer=tf.keras.optimizers.Adam,
        generator_optimizer_config={
            "learning_rate": 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.999
        },
        image_metrics={
            "PSNR": metrics.psnr,
            "PSNR_Y": metrics.psnr_on_y,
            "SSIM": metrics.ssim
        }
    )
    return model, config


def rrdb_model(pretrained_model=None):
    upsample_factor = 4
    config = ConfigUtil.training_config(
        train_data_paths=TRAIN_DATA,
        validation_data_path=VAL_DATA,
        results_save_path=SAVE_DIR,
        test_data_path=TEST_DATA,
        num_epochs=25, 
        batch_size=16,
        resize_filter=tf.image.ResizeMethod.BICUBIC,
        scale=upsample_factor,
        train_val_split=0,
        crop_imgs=True,
        crop_size=(128, 128, 3),
        num_crops=20,
        crop_naive=True,
        minimum_variation_patch=0.15,
        minimum_variation_batch=0.05,
        augmentations=[
            image_transforms.rotate90, image_transforms.flip_along_x,
            image_transforms.flip_along_y
        ],
        jpg_noise=False,
        jpg_noise_level=0,
        shuffle_buffer_size=4096,
        random_seed=None,
        early_stop_metric="PSNR",
        early_stop_patience=500
    )

    generator = Generator.rrdb(
        upsample_factor=upsample_factor,
        num_blocks=16,
        num_dense_blocks=3,
        num_convs=4,
        residual_scaling=0.2,
        pretrained_model=pretrained_model
    )
    model = SRModel.init(
        config,
        generator=generator,
        generator_optimizer=tf.keras.optimizers.Adam,
        generator_optimizer_config={
            "learning_rate": {
                "class_name": "PiecewiseConstantDecay",
                "config": {
                    "boundaries": [2e5, 4e5, 6e5],
                    "values": [2e-4, 1e-4, 5e-5, 2.5e-5]
                },
            },
            "beta_1": 0.9,
            "beta_2": 0.999
        },
        image_metrics={
            "PSNR": metrics.psnr,
            "PSNR_Y": metrics.psnr_on_y,
            "SSIM": metrics.ssim
        }
    )
    return model, config


def srgan_model(pretrained_model=None):
    """ inspired by SRGAN paper: https://arxiv.org/pdf/1609.04802.pdf """
    hr_dims = (96, 96)
    upsample_factor = 2
    config = ConfigUtil.training_config(
        train_data_paths=TRAIN_DATA,
        validation_data_path=VAL_DATA,
        results_save_path=SAVE_DIR,
        test_data_path=TEST_DATA,
        num_epochs=25,
        batch_size=8,
        resize_filter=tf.image.ResizeMethod.BICUBIC,
        scale=upsample_factor,
        train_val_split=0.1,
        crop_imgs=True,
        crop_size=(*hr_dims, 3),
        num_crops=20,
        augmentations=[
            image_transforms.rotate90, image_transforms.flip_along_x,
            image_transforms.flip_along_y
        ],
        jpg_noise=False,  
        jpg_noise_level=0,
        shuffle_buffer_size=4096,
        random_seed=None,
        early_stop_metric="PSNR",
        early_stop_patience=500
    )

    generator = Generator(
        upsample_factor=upsample_factor,
        architecture="srresnet",
        loss_functions=[
            VGGLoss(
                output_layers="block5_conv4",
                feature_scale=(1/12.75),
                loss_weight=1.0,
                total_variation_loss=False
            ),
            AdversarialLoss(
                weighted=True,
                loss_weight=1e-3
            )
        ],
        batch_norm=True,
        pretrained_model_path=None,
        pretrained_model=pretrained_model
    )

    discriminator = Discriminator.initialize_standard(
        weighted_loss=False, loss_weight=1.0,
        label_smoothing=False, smoothing_offset=0.3,
        input_dims=(96, 96)
    )

    model = SRModel.init(
        config,
        generator=generator,
        generator_optimizer=tf.keras.optimizers.Adam,
        generator_optimizer_config={
            "learning_rate": {
                "class_name": "PiecewiseConstantDecay",
                "config": {
                    "boundaries": [1e5],
                    "values": [1e-4, 1e-5]
                },
            },
            "beta_1": 0.9,
            "beta_2": 0.999
        },
        discriminator=discriminator,
        discriminator_optimizer=tf.keras.optimizers.Adam,
        discriminator_optimizer_config={
            "learning_rate": {
                "class_name": "PiecewiseConstantDecay",
                "config": {
                    "boundaries": [1e5],
                    "values": [1e-4, 1e-5]
                },
            },
            "beta_1": 0.9,
            "beta_2": 0.999
        },
    )
    return model, config


def esrgan_model(pretrained_model=None):
    """ inspired by ESRGAN paper: https://arxiv.org/abs/1809.00219 """
    hr_dims = (128, 128)
    upsample_factor = 2
    config = ConfigUtil.training_config(
        train_data_paths=TRAIN_DATA,
        validation_data_path=VAL_DATA,
        results_save_path=SAVE_DIR,
        test_data_path=TEST_DATA,
        num_epochs=25,
        batch_size=8,
        resize_filter=tf.image.ResizeMethod.BICUBIC,
        scale=upsample_factor,
        train_val_split=0,
        crop_imgs=True,
        crop_size=(*hr_dims, 3),
        num_crops=2,
        augmentations=[
            image_transforms.rotate90, image_transforms.flip_along_x,
            image_transforms.flip_along_y
        ],
        jpg_noise=False,
        jpg_noise_level=0,
        shuffle_buffer_size=4096,
        random_seed=None,
        early_stop_metric="PSNR",
        early_stop_patience=500
    )

    generator = Generator(
        upsample_factor=upsample_factor,
        architecture="rrdb",
        loss_functions=[
            MeanAbsoluteError(weighted=True, loss_weight=1e-2),
            RaAdversarialLoss(weighted=True, loss_weight=5e-3),
            VGGLoss(
                output_layers="block5_conv4",
                after_activation=False,
                feature_scale=1.0,
                loss_weight=1.0,
                total_variation_loss=False
            )
        ],
        num_blocks=16,
        num_dense_blocks=3,
        num_convs=4,
        residual_scaling=0.2,
        pretrained_model_path=None,
        pretrained_model=pretrained_model
    )

    discriminator = Discriminator.initialize_relativistic(
        weighted_loss=False, loss_weight=1.0,
        input_dims=hr_dims
    )

    model = SRModel.init(
        config,
        generator=generator,
        generator_optimizer=tf.keras.optimizers.Adam,
        generator_optimizer_config={
            "learning_rate": {
                "class_name": "PiecewiseConstantDecay",
                "config": {
                    "boundaries": [5e4, 1e5, 2e5, 3e5],
                    "values": [1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6]
                },
            },
            "beta_1": 0.9,
            "beta_2": 0.999
        },
        discriminator=discriminator,
        discriminator_optimizer=tf.keras.optimizers.Adam,
        discriminator_optimizer_config={
            "learning_rate": {
                "class_name": "PiecewiseConstantDecay",
                "config": {
                    "boundaries": [5e4, 1e5, 2e5, 3e5],
                    "values": [1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6]
                },
            },
            "beta_1": 0.9,
            "beta_2": 0.999
        },
    )
    return model, config


if __name__ == "__main__":
    # pre-train mse based network
    #m1, config = srresnet_model()
    m1, config = rrdb_model()
    pipeline = DataPipeline.from_config(config)
    training.run_training(config, pipeline, m1, plotting_interval=100)

    # use the pre-trained network as generator for gan setting
    #m2, config = srgan_model()
    m2, config = esrgan_model(m1.generator())
    pipeline = DataPipeline.from_config(config)
    training.run_training(config, pipeline, m2, plotting_interval=100)


