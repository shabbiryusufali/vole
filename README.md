<h1 align="center">VOLE (Vulnerability Observance and Learning-based Exploitation)</h1>

VOLE is a tool for detecting common bug classes in program binaries. It leverages:

- [angr](https://github.com/angr/angr) for symbolic execution, control-flow graph (CFG) recovery, and intermediate representation (IR) lifting
- [VEXIR2Vec](https://arxiv.org/abs/2312.00507) to derive vector embeddings of IR
- [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) for classification
- [NIST SARD Juliet](https://samate.nist.gov/SARD/test-suites/112) as training data

## Setting up VOLE for development

### Requirements

- Linux/WSL (macOS and Windows are unsupported)
- Python 3.11+
- pip

### Optional

> [!NOTE]
> If you do not have a CUDA or ROCm compatible GPU, you can skip the NVIDIA/AMD requirements.

- [Docker](https://www.docker.com/)
- [CUDA 12.9+ (NVIDIA GPUs)](https://developer.nvidia.com/cuda-downloads)
- [ROCm 6.4+ (AMD GPUs)](https://rocm.docs.amd.com/en/latest/) - While ROCm may work, it is not officially supported. Only use if you have ROCm pre-installed.

### Installation and Setup

#### 1. Activating Venv

```bash
python -m venv ./venv
source ./venv/bin/activate
```

#### 2. Installing Dependencies

```bash
python -m ensurepip
python -m pip install -r requirements.txt
python -m pip install -r requirements-nvidia.txt # (Optional) For NVIDIA GPUs
```

#### 3. Install Required Models and Data

```bash
python setup.py
```

#### 4. Compiling Training Data

> [!NOTE]
> For consistency, a Dockerfile has been provided to compile the SARD test cases

From the root directory of the repository:

1. (Optional) Build the Docker image with `docker build -t sard-env:latest .`
2. Compile the target CWEs per CWE-ID by running:
  a. Bare metal: `python vole/make.py CWE<ID> data/SARD`
  b. Docker: `docker run -it --rm -v "$PWD":/usr/src/env -w /usr/src/env sard-env python3 vole/make.py CWE<ID> data/SARD`

### Contributing Changes

1. Before starting work, ensure your local repo is up to date!
2. When assigned an issue, create a new branch for the issue by:
  1. Going to the "Development" tab
  2. Selecting "Create a branch"
  3. Clicking "Create branch" 
  4. Checking out the new branch locally
3. Do what you need to do
4. Before pushing changes, run `ruff format .` to format them (no ugly code, sorry)
