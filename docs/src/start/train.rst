Training Models
===============

| There are two ways to configure the components needed for training. You can either set up your configuration via YAML files or directly with python code.



Configure Training with YAML
----------------------------

A simple YAML configuration for training a resnet without an adversarial component could look like this:

.. code-block:: yaml

   general:
     operation: training
     train_data_paths:
       - ./data/datasets/reduced/div2k/8
     results_save_path: ./data/results
     num_epochs: 3
     batch_size: 8
     scale: 2
     train_val_split: 0.5
     crop_imgs: true
     crop_size: !!python/tuple [80, 80, 3]
     num_crops: 16
   
   model:
     generator:
       upsample_factor: 2
       architecture: srresnet
       loss_functions:
         - loss_function: MeanSquaredError
     generator_optimizer: Adam


To initialize the components and start the training you could just use the Makefile with (*Note: The yaml config needs to be inside ./examples/training/ for this to work*):

.. code-block:: bash

  make training_example TRAINING_CONFIG=name_of_yaml_config

If you don't want to use make or want to put your yaml outside the examples folder, you can start the training from python like this:


.. code-block:: python

  from simple_sr.utils.config.config_util import ConfigUtil
  from simple_sr.training import training_utils

  # my_training.py
  config, pipeline, model = ConfigUtil.from_yaml(config_yaml_path)
  training_utils.run_training(config, pipeline, model)

Now from the command-line start your file:

.. code-block:: bash

  # make sure your environment is activated
  source .env/bin/activate

  # start training
  python -m path.to.my_training


Configure Training with Python
------------------------------

A configuration with python code equivalent to the before shown YAML configuration would look like this:

.. code-block:: python

  import tensorflow as tf
  from simple_sr.utils.config.config_util import ConfigUtil
  from simple_sr.models.generator import Generator
  from simple_sr.models.sr_model import SRModel
  from simple_sr.data_pipeline.data_pipeline import DataPipeline
  from simple_sr.utils.models.loss_functions.mean_squared_error import MeanSquaredError
  from simple_sr.operations import training

  upsample_factor = 2
  config = ConfigUtil.training_config(
      train_data_paths="./data/datasets/div2k/8",
      num_epochs=3,
      batch_size=8,
      scale=upsample_factor,
      train_val_split=0.5,
      crop_imgs=True,
      crop_size=(80, 80, 3),
      num_crops=16,
  )

  generator = Generator(
      upsample_factor=upsample_factor,
      architecture="srresnet",
      loss_functions=[MeanSquaredError()]
  )

  model = SRModel.init(
      config,
      generator=generator,
      generator_optimizer=tf.keras.optimizers.Adam,
  )

  pipeline = DataPipeline.from_config(config)

  # Now that you have all components initialized you can start the training
  training.run_training(config, pipeline, model)


Again as before start your file like so:

.. code-block:: bash

  # make sure your environment is activated
  source .env/bin/activate

  # start training
  python -m path.to.my_training     # make sure to leave out the ".py" file ending
