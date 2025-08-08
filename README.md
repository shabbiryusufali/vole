# VOLE (Vulnerability Observance and Learning-based Exploitation)

## Setting up VOLE for development

### Requirements

- Linux/WSL (macOS and Windows are not supported)
- Python 3.11+
- pip

### Optional

> [!NOTE]
> Docker is required if your system does not support GCC 15.1 or later.

> [!NOTE]
> If you do not have a CUDA or ROCm compatible GPU, you can skip the NVIDIA/AMD requirements.

- [Docker](https://www.docker.com/)
- [CUDA 12.9+ (NVIDIA GPUs)](https://developer.nvidia.com/cuda-downloads)
- [ROCm 6.4+ (AMD GPUs)](https://rocm.docs.amd.com/en/latest/)

### Installation and Setup

#### 1. Venv

```bash
python -m venv ./venv
source ./venv/bin/activate
```

#### 2. Installing Dependencies

```bash
python -m ensurepip
python -m pip install -r requirements.txt
python -m pip install -r requirements-nvidia.txt # (Optional) For NVIDIA GPUs
python -m pip install -r requirements-amd.txt # (Optional) For AMD GPUs
```

#### 3. Setup the FastText model and SARD dataset

```bash
python setup.py
```

#### 4. Acquiring Training Data

> [!NOTE]
> For consistent results, compile the training data with GCC 15.1
> A Dockerfile has been supplied to ensure a reproducible environment

1. Ensure you are in the root directory of the repository
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
4. Before pushing changes, run `ruff format .` to format them (no ugly code, sorry)
