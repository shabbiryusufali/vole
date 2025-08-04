# cmpt-479-project

## Setting up VOLE for development

### Requirements

- [PyPy](https://pypy.org)
- [Rust](https://www.rust-lang.org/)
- pip

### Optional

- Docker

### Installation and Setup

#### 1. PyPy

1. Navigate to https://pypy.org/download.html
2. Download the latest precompiled binary for your platform
3. Extract the binary from the archive (`tar xz <pypy-version>.tar.bz2`)
4. Add a symlink from `/usr/local/bin/pypy` to `path/to/pypy-version/bin/pypy` (`ln -s path/to/pypy-version/bin/pypy /usr/local/bin/pypy`)
  1. Alternatively, add PyPy to `PATH` by modifying `~/.bashrc` (`export PATH="$PATH:path/to/pypy-version/bin"`)

#### 2. Rust

> [!note]
> Rust is required to build Angr's dependencies

1. Navigate to https://forge.rust-lang.org/infra/other-installation-methods.html
2. Follow the provided instructions for your platform

#### 2. Venv

```bash
pypy -m venv ./venv
source ./venv/bin/activate
```

#### 3. Installing Dependencies

```bash
pypy -m ensurepip
pypy -m pip install -r requirements.txt
```

#### 4. Acquiring Training Data

> [!NOTE]
> For consistent results, compile the training data with GCC 15.1
> A Dockerfile has been supplied to ensure a reproducible environment

1. Run `./data/SARD/download.sh`
2. (Optional) Build the Docker image with `docker build -t vole-env:latest .`
3. Compile the target CWEs per CWE-ID by running:
  a. Bare metal: `python vole/make.py <CWE-ID> data/SARD`
  b. Docker: `docker run -it --rm -v "$PWD":/usr/src/env -w /usr/src/env vole-env python3 vole/make.py CWE<ID> data/SARD`

### Contributing Changes

1. Before starting work, ensure your local repo is up to date!
2. When assigned an issue, create a new branch for the issue by:
  1. Going to the "Development" tab
  2. Selecting "Create a branch"
  3. Clicking "Create branch" 
  4. Checking out the new branch locally
3. Do what you need to do
4. Before pushing changes, run `ruff format` to format them (no ugly code, sorry)
