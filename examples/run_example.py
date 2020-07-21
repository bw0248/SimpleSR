import sys
import os
import time
from pathlib import Path
import ruamel.yaml
from simple_sr.utils.config.config_util import ConfigUtil
from simple_sr.operations import training, evaluation


def parse_operation_from_yaml(config_yaml_path):
    if not os.path.isfile(config_yaml_path):
        raise ValueError("could not locate config.yaml")
    with open(config_yaml_path) as f:
        conf_yaml = ruamel.yaml.load(f)
    if conf_yaml["general"]["operation"] == "training":
        run_training_example(config_yaml_path)
    elif conf_yaml["general"]["operation"] == "evaluation":
        run_evaluation_example(config_yaml_path)
    elif conf_yaml["general"]["operation"] == "inference":
        run_inference_example(config_yaml_path)
    else:
        raise ValueError("operation in config.yaml not understood - choose one of [training, evaluation]")


def run_training_example(config_yaml_path):
    print(f"running training example with {Path(config_yaml_path).stem}")
    config, pipeline, model = ConfigUtil.from_yaml(config_yaml_path)
    training.run_training(config, pipeline, model)


def run_inference_example(config_yaml_path):
    print(f"running evaluation example with {Path(config_yaml_path).stem}")
    config, pipeline = ConfigUtil.from_yaml(config_yaml_path)
    start = time.perf_counter()
    evaluation.evaluate_on_testdata(
        config=config, save_single=config.save_single,
        grid=config.grid, interpolate=config.interpolate,
        with_original=config.with_original, combine_halfs=config.combine_halfs,
        pipeline=pipeline
    )
    print(f"duration: {time.perf_counter() - start}")


def run_evaluation_example(config_yaml_path):
    print(f"running evaluation example with {Path(config_yaml_path).stem}")
    config, pipeline = ConfigUtil.from_yaml(config_yaml_path)
    evaluation.evaluate_on_validationdata(
        config=config, pipeline=pipeline, save_grid=config.grid,
        combine_halfs=config.combine_halfs, save_single=config.save_single
    )


def raise_error(msg):
    raise ValueError(f"{msg} - please supply an example config.yaml from ./examples")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise_error("no config.yaml path supplied")
    elif len(sys.argv) > 2:
        raise_error("received more arguments than expected")
    parse_operation_from_yaml(sys.argv[1])


