.PHONY: setup build clean rebuild lint test

# Configuration
VENV_DIR := .venv
PYTHON := python3.11
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python

# Main targets
all: setup build

# Create virtual environment with Python 3.11 and install dependencies
setup:
	@echo "Creating virtual environment with Python 3.11..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "Installing/upgrading dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install ruff scikit-learn scipy numpy build wheel

# Build the C++ extension module
build:
	@echo "Building decision_tree_classifier package..."
	$(PYTHON_VENV) -m build
	@echo "Installing package..."
	@WHEEL_FILE=$$(find dist -name "*.whl" -type f 2>/dev/null | head -1) && \
	if [ -n "$$WHEEL_FILE" ]; then \
		echo "Installing $$WHEEL_FILE"; \
		$(PIP) install --force-reinstall $$WHEEL_FILE; \
	else \
		echo "No wheel file found in dist/"; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf _skbuild
	rm -rf decision_tree_classifier/build
	rm -rf decision_tree_classifier/dist
	rm -rf decision_tree_classifier/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.so" -delete
	find . -type f -name "*.o" -delete

# Rebuild from scratch
rebuild:
	@echo "Creating fresh build environment..."
	$(MAKE) clean
	rm -rf $(VENV_DIR)
	$(MAKE) setup
	$(MAKE) build

lint:
	@echo "Linting code with ruff..."
	$(VENV_DIR)/bin/ruff format .
	$(VENV_DIR)/bin/ruff check . --fix

run:
	@echo "Running example..."
	$(PYTHON_VENV) main.py
