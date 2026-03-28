# Contributing to helix

Thanks for your interest in contributing to helix!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

Once the Python virtual environment is set up, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we follow the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static analysis,
and [mypy](https://mypy.readthedocs.io/en/stable/) for type checking. The pre-commit
hooks will catch errors before you submit a PR.
