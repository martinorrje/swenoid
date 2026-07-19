# Contributing

Swenoid is a research codebase that also drives physical hardware. Changes are
welcome, but they need to be reproducible and easy to review. Keep pull requests
focused, explain the reason for the change, and leave unrelated cleanup for a
separate patch.

Use a GitHub issue for bugs, proposals, and questions that should have a public,
durable answer. Small fixes do not need an issue first. Please open one before
work that changes a public CLI, an observation or action contract, a motion-file
format, an actuator model, or the hardware protocol.

A useful bug report includes the release or commit, operating system, hardware or
simulation backend, exact command, full traceback or output, and the smallest
reproduction you can provide. Do not put credentials or details that could lead
to unsafe robot motion in a public issue; send those to `morrje@kth.se`.

## Set up a development checkout

Swenoid uses Python 3.12 or 3.13 and
[`uv`](https://docs.astral.sh/uv/) for dependency management. From a fresh
clone:

```bash
git clone https://github.com/martinorrje/swenoid.git
cd swenoid
uv sync --frozen
```

The default development group contains the dependencies used by CI. Run tools
through `uv run` so they use the locked environment.

Create a branch from an up-to-date `main` branch. During development, format and
lint the files you touch:

```bash
uv run ruff format
uv run ruff check .
```

## Patch requirements

- Add or update tests whenever behavior changes. A regression test is usually
  the clearest description of a bug fix.
- Preserve the documented 50 Hz control loop, joint ordering, policy I/O, and
  motion-data contracts unless the pull request deliberately changes them.
- Update the README or the relevant file under `docs/` when a command, config
  field, data format, or user-visible default changes.
- Keep new dependencies narrow. Add them to the correct optional extra with
  `uv add --optional <extra> <package>`, or to the development group with
  `uv add --dev <package>`. Commit `uv.lock` whenever the dependency graph
  changes.
- Do not mix generated files, wholesale formatting, or mechanical renames into
  a functional change unless they are required for it.

Do not commit raw BONES-SEED data, retargeted motion files, credentials, W&B
caches, local calibration profiles, checkpoints, ONNX exports, or other
generated model artifacts. Reviewed model binaries belong in GitHub release
assets, with their provenance and checksums, rather than in the source tree.

## Run the checks

Before opening a pull request, run the same checks as CI:

```bash
uv run --frozen ruff format --check
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest -q
uv build
```

For a small change, run the closest test while iterating, then run the complete
suite before submitting:

```bash
uv run pytest tests/test_deployment.py -q
uv run pytest tests/test_deployment.py::test_name -q
```

If a check cannot run on your machine, say which one and why in the pull request.
Do not silently omit it.

## Hardware-facing changes

Code that can move the robot must retain the existing safeguards: torque-off
initialization, target and joint-limit checks, rejection of non-finite values,
bounded communication retries, and torque-off shutdown. Unit tests with protocol
fakes are required even when the change has been exercised on hardware.

In the pull request, record the controller, firmware, bus configuration,
calibration profile, physical support, emergency-stop arrangement, and exact
test procedure. Never commit a calibration profile for an individual robot.
Follow the bring-up checklist in [`docs/deployment.md`](docs/deployment.md) and
test the exported policy in sim2sim before enabling motor power.

## Pull requests

A useful pull request description answers four questions:

1. What problem does this solve?
2. What changed, and what did not?
3. How was it tested?
4. Does it alter a public interface, experiment result, dataset boundary, or
   hardware-safety assumption?

Prefer small commits with imperative subjects. Reviewers should be able to
understand each commit without reverse-engineering a later fixup. Screenshots,
plots, short recordings, and benchmark numbers are useful when they demonstrate
behavior that tests cannot capture.

Contributions are accepted under the repository [MIT License](LICENSE). Submit
only work that you have the right to license.
