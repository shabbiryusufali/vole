# cmpt-479-project

## Setting up VOLE for development

### Requirements

- [PyPy](https://pypy.org)
- [Rust](https://www.rust-lang.org/)
- pip

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

1. Navigate to https://samate.nist.gov/SARD/test-suites/112
2. Download the archive and extract its contents in the `data/SARD` directory
3. Compile the target CWEs by CWE-ID by running `python scripts/make.py <CWE-ID> data/SARD` where `<CWE-ID>` is the literal string "CWE" followed by the numeric identifier (e.g. `CWE123`)
4. Lift the IR for these target CWEs by CWE-ID by running `python scripts/rip.py <CWE-ID> data/SARD`

### Contributing Changes

1. Before starting work, ensure your local repo is up to date!
2. When assigned an issue, create a new branch for the issue by:
  1. Going to the "Development" tab
  2. Selecting "Create a branch"
  3. Clicking "Create branch" 
  4. Checking out the new branch locally
3. Do what you need to do
4. Before pushing changes, run `ruff format` to format them (no ugly code, sorry)
