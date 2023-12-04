# Project Setup Instructions

## 1. Clone the Repository

To start, clone this repository to your local machine:

```bash
$ git clone https://github.com/YangWu1227/chicago-cta-ridership.git
```

## 2. Setting Up with `poetry`

There are two primary methods to set up and run `poetry` for this project:

### Method 1: Using the Official `poetry` Installer

1. Install `poetry` using the official installer for your operating system. Detailed instructions can be found at [Poetry's Official Documentation](https://python-poetry.org/docs/#installation). Make sure to add `poetry` to your PATH. Refer to the official documentation linked above for specific steps for your operating system.

2. Navigate to the root of the cloned project, which contains the `poetry.lock` and `pyproject.toml` files:

```bash
$ cd [path_to_cloned_repository]
```

3. Configure `poetry` to create the virtual environment inside the project's root directory:

```bash
$ poetry config --local virtualenvs.in-project true
```

4. Ensure Python `3.10` is installed on your system and install the project dependencies:

```bash
$ poetry install
```

### Method 2: Using `conda` and `poetry` Together

1. Create a new conda environment named `ts_env` with Python `3.10`:

```bash
$ yes | conda create --name ts_env python=3.10
```

2. Install `poetry` within the `ts_env` environment:

```bash
$ conda activate ts_env
$ pip3 install poetry
```

1. Navigate to the project root:

```bash
$ cd [path_to_cloned_repository]
```

5. Install the project dependencies (ensure that the `conda` environment is activated):

```bash
$ conda activate ts_env
$ poetry install
```

## [Optional] Automating Setup on MacOS (Intel chip `x86_64`)

If you are on a macOS machine with an Intel chip (`x86_64`), you can adapt the bash scripts located in the `scripts` directory to automate the `conda` setup process. Specifically, the `~/opt/anaconda3/bin/activate` path may need to be updated to the installation path on your system. 