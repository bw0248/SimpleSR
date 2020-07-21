.. _install-doc:

Installation
============

First obtain the code from `github <https://github.com/bw0248/SimpleSR>`_.

.. code-block:: bash

  git clone https://github.com/bw0248/SimpleSR
  cd SimpleSR


Running natively
----------------

If you want to run natively on your own machine set up a Python virtual environment and install the requirements.
You can either do this manually by invoking:

.. code-block:: bash

  python3.6 -m venv .env
  source .env/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt

Or you can use the supplied make file and things will be initialized for you.

.. code-block:: bash

  python3.6 -m venv .env
  make init

Running inside docker container
-------------------------------

| Another option is to run inside a docker container. The docker image can be build with the supplied `Dockerfile`.    

.. note:: The Dockerfile is based on a nvidia-cudnn image which is not particular lightweight, but will be needed for training on GPUs. If you don't plan on using a GPU you can alter the `Dockerfile` and inherit from another base image (Ubuntu for instance). Training on a GPU is definitely recommended though.

| Make sure you have docker installed and running.
| To build the image and start the container (depending on your docker installation you may need to run the following commands as root):

.. code-block:: bash

  # build image
  docker build -t simple_sr .

  # make sure the image was created successfully 
  docker images     # you should see an image called simple_sr

  # obtain image id (can also be done manually, by checking third column of 'docker images')
  img_id=$(docker images | grep simple_sr | awk '{print $3}')

  # start and enter container (container will stop automatically once you exit it)
  docker run -p 6006:6006 -it $img_id /bin/bash

  # you should now be inside the container
  # check that everything worked and requirements are installed
  cd dev/
  source .env/bin/activate
  pip list


To check that tensorboard (and everything else) is working correctly you can start the minimal training example.

.. note:: This will train a model on a tiny dataset containing only 8 images, so results will naturally be very bad. This is just to see if everything works.

.. code-block:: bash

    # assuming you're still inside the docker container...
    # start tensorboard server with supplied Makefile (or otherwise if you like)
    make tensorboard &

    # start minimal training example 
    make training_example TRAINING_CONFIG=minimal_example.yaml

If you now navigate to http://localhost:6006 in your browser you should see tensorboard and after a short while stats from your running training session.
