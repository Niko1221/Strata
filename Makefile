# Everyday launches after ./setup.sh. Override PYTHON or CONFIG when needed.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,python3))
CONFIG ?= strata.yaml

.PHONY: help init models check run
.DEFAULT_GOAL := help

help:
	@echo "make init    Create a local launch config (keeps an existing file)"
	@echo "make models  List installed model names"
	@echo "make check   Validate the config and local files without loading the GPU"
	@echo "make run     Run in the foreground; Ctrl+C stops the model"
	@echo "Use CONFIG=path/to/launch.yaml or PYTHON=path/to/python to override defaults."

init models check run:
	@"$(PYTHON)" run.py $@ --config "$(CONFIG)"
