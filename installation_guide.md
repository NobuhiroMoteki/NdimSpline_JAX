# Installation Guide

This guide describes how to set up an execution environment for **NdimSpline_JAX** from scratch on:

- **Part A:** Windows machine using WSL2 (Ubuntu)
- **Part B:** Native Linux machine (Ubuntu)

Both paths converge on the same Python environment managed by [`uv`](https://docs.astral.sh/uv/).

---

## Prerequisites (common)

- A working internet connection
- Administrator (`sudo`) privileges
- ~2 GB of free disk space

The library runs on **CPU by default**. GPU usage requires a CUDA-enabled `jaxlib`, which is out of scope for this guide.

---

## Part A: Windows + WSL2 (Ubuntu)

### A-1. Install WSL2 and Ubuntu

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu
```

Reboot if requested. After reboot, an Ubuntu terminal will open and prompt for a UNIX username and password. Set them.

Verify:

```powershell
wsl --status
wsl --list --verbose
```

Both should report **Default Version: 2** and that Ubuntu is **Running**.

### A-2. Open the Ubuntu shell

From the Start menu, launch "Ubuntu". All subsequent commands are executed inside this shell. Continue from [Common steps](#common-steps-from-ubuntu-shell).

---

## Part B: Native Linux (Ubuntu)

### B-1. Verify Ubuntu version

```bash
lsb_release -a
```

Ubuntu 22.04 LTS or newer is recommended.

### B-2. Open a terminal

Continue from [Common steps](#common-steps-from-ubuntu-shell).

---

## Common steps (from Ubuntu shell)

### 1. Update the package index and install build essentials

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates
```

### 2. Install `uv` (Python package & environment manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell so that `uv` is on `PATH`:

```bash
source ~/.bashrc
uv --version
```

Expected output: `uv 0.x.y` or newer.

> **Why `uv`?** It manages Python interpreters and per-project virtual environments without polluting the system Python, and is significantly faster than `pip`.

### 3. Clone the repository

```bash
cd ~
git clone https://github.com/NobuhiroMoteki/NdimSpline_JAX.git
cd NdimSpline_JAX
```

### 4. Create a virtual environment with the required Python version

`NdimSpline_JAX` requires **Python ≥ 3.10**. The default development version is 3.13.

```bash
uv venv --python 3.13
```

This creates a `.venv/` directory in the project root.

### 5. Install runtime and test dependencies

```bash
uv pip install jax numpy scipy pytest
```

Verify versions:

```bash
.venv/bin/python -c "import jax, numpy, scipy, pytest; \
  print('jax', jax.__version__); \
  print('numpy', numpy.__version__); \
  print('scipy', scipy.__version__); \
  print('pytest', pytest.__version__)"
```

### 6. Run the test suite

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: **42 passed** in ~30 seconds. If all tests pass, the installation is correct.

### 7. Run the demo

```bash
.venv/bin/python caller.py
```

This computes a 5D spline interpolant and prints the value, gradient, and benchmark timings.

---

## Optional: Jupyter Notebook support

To run the dimension-specific notebooks under `jupyter_notebooks/`:

```bash
uv pip install jupyterlab ipykernel
.venv/bin/python -m ipykernel install --user --name ndimsplinejax --display-name "NdimSpline_JAX (.venv)"
.venv/bin/jupyter lab
```

In Jupyter, select the **NdimSpline_JAX (.venv)** kernel for each notebook.

---

## Optional: VS Code integration

1. Install [VS Code](https://code.visualstudio.com/) (Windows) and the **WSL** extension (Windows-only) and **Python** extension.
2. From the Ubuntu shell, run:

   ```bash
   cd ~/NdimSpline_JAX
   code .
   ```

3. When VS Code opens, choose **`.venv/bin/python`** as the interpreter (Command Palette → "Python: Select Interpreter").

---

## Optional: Building the technical note PDF

The LaTeX source lives in `docs/theory_note.tex`. To rebuild the PDF locally:

```bash
sudo apt-get install -y texlive-latex-recommended texlive-fonts-recommended \
                        texlive-latex-extra texlive-science latexmk
cd docs && latexmk -pdf theory_note.tex
```

The `physics`, `siunitx`, `booktabs`, and `bm` packages are required (provided by the `texlive-*` packages above).

---

## Troubleshooting

### `uv: command not found` after installation

The `uv` installer adds `~/.local/bin` to `PATH` via `~/.bashrc`. If `source ~/.bashrc` does not work, log out and back in, or add the directory manually:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### `An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.`

This warning is harmless — JAX falls back to CPU automatically, which is the supported configuration for this project.

### `apt_pkg` ModuleNotFoundError when running `apt-get`

If the system `python3` symlink has been redirected (e.g., to a non-default version such as 3.13 from `deadsnakes`), restore the Ubuntu default:

```bash
sudo ln -sf /usr/bin/python3.10 /usr/bin/python3
```

Then re-run `sudo apt-get update`.

### Tests fail with `ModuleNotFoundError: No module named 'jax'`

Ensure you are using the venv Python (`.venv/bin/python`), not the system `python3`.

---

## Uninstall

To remove the project entirely:

```bash
cd ~ && rm -rf ~/NdimSpline_JAX
```

The `.venv/` directory is inside the project, so this removes it too.
