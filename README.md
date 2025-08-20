# Machine Learning with Python

A repository for machine learning application and learning. 

This is a **loosely structured, dump-all** repository that includes notebooks, projects, scripts, and files, focusing on a wide range of machine learning tasks, experiments, and topics.

## Dependencies

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage its dependencies. The required python version in `pyproject.toml` is `>=3.12`.

### Python Interpreter managed by `uv`

```bash
$ uv sync --frozen --all-groups
```

### Python Interpreter managed by `conda`

```bash
$ conda create --name python_ml -y python=3.12
$ uv sync --frozen --all-groups
```

### Python Interpreter managed by `pyenv`

```bash
# List available Python versions
$ pyenv install --list | grep " 3\.\(12\)\."
# As an example, install Python 3.12.8
$ pyenv install 3.12.8
$ pyenv local 3.12.8
$ uv sync --frozen --all-groups
```
