# Contributing

Thank you for improving Swenoid. Please use GitHub issues for reproducible bugs,
feature proposals, and questions that need a lasting public answer.

## Development setup

Install `uv`, clone the repository, and create the locked development
environment:

```bash
uv sync --frozen
```

Before opening a pull request, run the same checks as CI:

```bash
uv run --frozen ruff format --check
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest -q
uv build
```

Keep changes focused, add tests for behavior changes, and update user-facing
documentation when commands or policy contracts change. Do not commit raw
BONES-SEED data, retargeted motion files, credentials, W&B caches, robot-specific
calibration files, or generated model artifacts.

## Hardware changes

Changes that can move hardware must preserve torque-off initialization, limit
checks, non-finite target rejection, and shutdown behavior. Describe the
physical test setup and safety precautions in the pull request. Unit tests with
protocol fakes are required even when hardware-in-the-loop testing was used.

By contributing, you agree that your contribution is licensed under the MIT
License and that you have the right to submit it.

