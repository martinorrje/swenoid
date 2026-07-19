# Sim2Sim CLI Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing sim2sim command select the newest ONNX checkpoint and launch MuJoCo correctly on macOS.

**Architecture:** Keep W&B selection inside the shared policy downloader and add one small preflight helper to the sim2sim entry point. Preserve explicit checkpoint selection and all non-macOS behavior.

**Tech Stack:** Python 3.12, W&B Public API, MuJoCo `mjpython`, pytest

---

### Task 1: Select the newest W&B ONNX file

**Files:**
- Modify: `tests/test_deployment.py`
- Modify: `src/swenoid/deployment/policy.py`

- [ ] **Step 1: Write failing downloader tests**

Add lightweight fake W&B run and file objects. Test that `download_wandb_onnx`
selects the greatest `(updated_at, name)` pair, honors an exact filename, and
raises clear errors for a missing explicit name and an empty ONNX list.

- [ ] **Step 2: Run the downloader tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_deployment.py -k wandb
```

Expected: newest-selection assertions fail because the current downloader
rejects multiple ONNX files.

- [ ] **Step 3: Implement minimal W&B selection**

Change `download_wandb_onnx` to retain W&B file objects, filter exact names when
requested, reject zero matches, and otherwise select:

```python
selected = max(candidates, key=lambda item: (item.updated_at or "", item.name))
```

Download `selected` into the existing run-specific cache only when absent.

- [ ] **Step 4: Run downloader tests and verify GREEN**

Run:

```bash
.venv/bin/pytest -q tests/test_deployment.py -k wandb
```

Expected: all W&B downloader tests pass.

### Task 2: Relaunch sim2sim through `mjpython` on macOS

**Files:**
- Modify: `tests/test_deployment.py`
- Modify: `src/swenoid/deployment/sim2sim.py`

- [ ] **Step 1: Write failing launcher tests**

Test a new `_ensure_mjpython` helper with monkeypatched platform, executable,
MuJoCo dispatcher, and `os.execv`. Cover the forwarded command:

```python
[
    "/venv/bin/mjpython",
    "-m",
    "swenoid.deployment.sim2sim",
    *original_arguments,
]
```

Also verify it is a no-op outside macOS and when the MuJoCo dispatcher is
already active, and that a missing launcher raises a direct `RuntimeError`.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_deployment.py -k mjpython
```

Expected: test collection fails because `_ensure_mjpython` does not exist.

- [ ] **Step 3: Implement minimal macOS preflight**

Add `_ensure_mjpython` to `sim2sim.py`. On Darwin with no active
`mujoco.viewer._MJPYTHON`, find `mjpython` beside `sys.executable`, validate it,
and call `os.execv` with the module name and original arguments. Call the helper
as the first statement in `main`.

- [ ] **Step 4: Run launcher tests and verify GREEN**

Run:

```bash
.venv/bin/pytest -q tests/test_deployment.py -k mjpython
```

Expected: all `mjpython` tests pass.

### Task 3: Verify the complete repair

**Files:**
- Modify if required by formatting only: `tests/test_deployment.py`
- Modify if required by formatting only: `src/swenoid/deployment/policy.py`
- Modify if required by formatting only: `src/swenoid/deployment/sim2sim.py`

- [ ] **Step 1: Run focused deployment tests**

```bash
.venv/bin/pytest -q tests/test_deployment.py
```

Expected: all deployment tests pass.

- [ ] **Step 2: Run formatting and lint checks**

```bash
.venv/bin/ruff format --check src/swenoid/deployment/policy.py src/swenoid/deployment/sim2sim.py tests/test_deployment.py
.venv/bin/ruff check src/swenoid/deployment/policy.py src/swenoid/deployment/sim2sim.py tests/test_deployment.py
```

Expected: both commands exit successfully.

- [ ] **Step 3: Run type checking and the full test suite**

```bash
.venv/bin/pyright src/swenoid/deployment/policy.py src/swenoid/deployment/sim2sim.py
.venv/bin/pytest -q
```

Expected: type checking reports no errors and the full suite passes.

- [ ] **Step 4: Review the final diff**

```bash
git diff --check
git diff -- src/swenoid/deployment/policy.py src/swenoid/deployment/sim2sim.py tests/test_deployment.py
```

Expected: no whitespace errors and only the approved behavior and regression
tests are present.
