PYTHON ?= python3

.PHONY: install install-dev test lint typecheck render train download-models

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,train,edge]"

test:
	pytest

lint:
	ruff check glyphcast tests

typecheck:
	mypy glyphcast

render:
	$(PYTHON) -m glyphcast.cli render giphy.gif --mode terminal

train:
	$(PYTHON) -m glyphcast.cli train-chars

download-models:
	$(PYTHON) -m glyphcast.cli download-models
