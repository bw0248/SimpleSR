import sys
import ruamel.yaml

from simple_sr.utils.models.loss_functions.mean_absolute_error import MeanAbsoluteError
from simple_sr.utils.models.loss_functions.mean_squared_error import MeanSquaredError
from simple_sr.utils.models.loss_functions.adversarial_loss import AdversarialLoss
from simple_sr.utils.models.loss_functions.ra_adversarial_loss import RaAdversarialLoss
from simple_sr.utils.models.loss_functions.vgg_loss import VGGLoss
from simple_sr.utils.models.loss_functions.discriminator_loss import DiscriminatorLoss
from simple_sr.utils.models.loss_functions.ra_discriminator_loss import RaDiscriminatorLoss
from simple_sr.utils.image import image_transforms

YAML_GENERAL_KEY = "general"
YAML_AUGMENTATION_KEY = "augmentations"
YAML_RESIZE_FILTER_KEY = "resize_filter"
YAML_MODEL_KEY = "model"
YAML_GENERATOR_KEY = "generator"
YAML_GENERATOR_OPTIMIZER_KEY = "generator_optimizer"

AUGMENTATION_MODULE_PATH = "simple_sr.utils.image.image_transforms"


def prepare_for_training_config(config_yaml):
    return init_augmentations(config_yaml)


def swap_key(config_yaml, yaml_key, to_swap):
    config_yaml[yaml_key] = to_swap
    return config_yaml


def prepare_for_evaluation_config(config_yaml):
    if config_yaml[YAML_GENERAL_KEY][YAML_RESIZE_FILTER_KEY] is not None:
        resize_filter = string_to_lib_object(
            "tensorflow",
            ["image", "ResizeMethod",
             config_yaml[YAML_GENERAL_KEY][YAML_RESIZE_FILTER_KEY].upper()]
        )
        config_yaml[YAML_GENERAL_KEY][YAML_RESIZE_FILTER_KEY] = resize_filter
    return config_yaml


def init_loss_functions_from_yaml(config_yaml):
    initialized_loss_funcs = list()
    for loss_func in config_yaml["loss_functions"]:
        obj = getattr(sys.modules[__name__], loss_func["loss_function"])
        params = dict(filter(lambda elem: elem[0] != "loss_function", loss_func.items()))
        initialized_loss_funcs.append(obj(**params))
    return initialized_loss_funcs


def string_to_lib_object(lib, modules):
    lib = __import__(lib)
    for module in modules:
        lib = getattr(lib, module)
    return lib


def init_augmentations(config_yaml):
    augmentations = list()
    if YAML_AUGMENTATION_KEY not in config_yaml[YAML_GENERAL_KEY] \
            or config_yaml[YAML_GENERAL_KEY][YAML_AUGMENTATION_KEY] is None:
        return config_yaml

    for augmentation in config_yaml[YAML_GENERAL_KEY][YAML_AUGMENTATION_KEY]:
        augmentation_func = getattr(
            sys.modules[AUGMENTATION_MODULE_PATH], augmentation
        )
        augmentations.append(augmentation_func)

    # switch out strings with actual functions
    config_yaml[YAML_GENERAL_KEY][YAML_AUGMENTATION_KEY] = augmentations
    return config_yaml


def load_yaml(config_yaml_path):
    if type(config_yaml_path) is not dict:
        with open(config_yaml_path) as f:
            config_yaml = ruamel.yaml.load(f)
    else:
        # yaml seems to be already loaded
        config_yaml = config_yaml_path
    return config_yaml


if __name__ == "__main__":
    pass


