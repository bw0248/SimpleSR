TB_LOG_DIR="./data/results/training"
IP=$(shell hostname -I)
PORT=6006
ENV_NAME=".env"
EXAMPLE_CONFIG_FOLDER="./examples"
TRAINING_EXAMPLES_FOLDER="$(EXAMPLE_CONFIG_FOLDER)/training"
EVAL_EXAMPLES_FOLDER="$(EXAMPLE_CONFIG_FOLDER)/evaluation"
TRAINING_CONFIG?="minimal_example.yaml"
EVALUATION_CONFIG?="evaluation_example.yaml"

init: venv
	. $(ENV_NAME)/bin/activate; pip install --upgrade pip; pip install -r requirements.txt
	@echo "init done"

shell:
	. $(ENV_NAME)/bin/activate; python

venv:
	python3.8 -m venv $(ENV_NAME)

training_example:
	. $(ENV_NAME)/bin/activate; python -m examples.run_example $(TRAINING_EXAMPLES_FOLDER)/$(TRAINING_CONFIG)

evaluation_example:
	. $(ENV_NAME)/bin/activate; python -m examples.run_example $(EVAL_EXAMPLES_FOLDER)/$(EVALUATION_CONFIG)

tests: init
	. $(ENV_NAME)/bin/activate; python -m tests.test_suite
	@echo "tests completed"

tensorboard:
	@echo starting server in folder $(TB_LOG_DIR) on host $(IP):$(PORT)
	$(shell . $(ENV_NAME)/bin/activate; tensorboard --logdir $(TB_LOG_DIR) --host $(IP) --port $(PORT))

clean:
	rm -rf $(ENV_NAME) 2>/dev/null

.PHONY: init shell venv training_example evaluation_example tests tensorboard clean