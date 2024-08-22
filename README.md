# Machine Learning with Python

A repository for machine learning application and learning. 

This is a **loosely structured, dump-all** repository that includes notebooks, projects, scripts, and files, accommodating a wide range of machine learning tasks, experiments, and projects. It’s designed without strict organization or dependency management, providing the flexibility to explore and learn across various topics.

## Set Up

The dependencies are managed by [pdm](https://github.com/pdm-project/pdm) and [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html):

```shell
$ yes | conda create --name python_for_machine_learning python=3.11.9
$ conda activate python_for_machine_learning
# Use the conda-installed python interpreter
$ pdm use $(which python3)
$ pdm install --prod
```

However, any other dependency management tool can be used instead. Not all dependencies are explicitly declared in the pyproject.toml file. 
