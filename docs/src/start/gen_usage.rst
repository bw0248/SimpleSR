General Principles
==================

| Generally there are two things you can do with SimpleSR. You can either train models on a dataset, or you can evaluate already trained models on your test data.

For training you will need to:

- Obtain a set of training images (links to some popular Super-Resolution data sets can be found `here <https://github.com/bw0248/SimpleSR#datasets>`_.
- Initialize a :code:`DataPipeline` with your training data
- Initialize an :code:`SRModel`
  - The :code:`SRModel` needs a :code:`Generator` - this the component that learns to upscale images
  - Optionally you can also provide a :code:`Discriminator` to the :code:`SRModel` to train in adversarial/GAN mode
- finally you need to start the training process with the initiaized :code:`DataPipeline` and :code:`SRModel`

For evaluation you will need:

- A :code:`DataPipeline` initialized with your test images (initialized with :code:`evaluate_only=True`)
- The path to a saved model

See the examples folder for some training and evaluation examples and read the docs for training/evaluating models for further info.
