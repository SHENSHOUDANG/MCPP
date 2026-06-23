# Environment Snapshot Before GPU Rebuild

Date: 2026-06-23
Workspace: `D:\projects\GAT-MAPPO\MCPP分支-巡检`

## Project Dependency Sources

- `pyproject.toml`
  - Python: `>=3.10`
  - Runtime packages: `numpy>=1.24`, `torch>=2.0`, `tensorboard>=2.10`, `matplotlib>=3.7`, `tomli>=2.0` for Python older than 3.11.
- `environment.yml`
  - Conda CPU environment: Python 3.10, PyTorch 2.2, `cpuonly`, NumPy 1.26, Matplotlib, pip, tomli.
- `environment.cuda.yml`
  - Conda CUDA environment: Python 3.10, PyTorch >=2.7, `pytorch-cuda=12.8`, NumPy 1.26, Matplotlib, TensorBoard, pip, tomli.

## Source Import Check

Third-party imports used by project code and tests:

- `torch`
- `numpy`
- `matplotlib`
- `tensorboard` through training logs and README workflow
- `tomli` fallback for Python versions below 3.11

The remaining imports are standard library modules or local `mathbased_mcpp` modules.

## Deleted Local Environments

### `.venv`

- Python: 3.10.11
- PyTorch: `2.2.2+cpu`
- `torch.cuda.is_available()`: `False`
- `torch.version.cuda`: `None`
- `torch.cuda.device_count()`: `0`

Installed packages before deletion:

```text
absl-py==2.4.0
contourpy==1.3.2
cycler==0.12.1
filelock==3.29.1
fonttools==4.63.0
fsspec==2026.4.0
grpcio==1.81.0
Jinja2==3.1.6
kiwisolver==1.5.0
Markdown==3.10.2
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
networkx==3.4.2
numpy==1.26.4
packaging==26.2
pillow==12.2.0
protobuf==7.35.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tomli==2.4.1
torch==2.2.2+cpu
typing_extensions==4.15.0
Werkzeug==3.1.8
```

### `.venv-cuda`

This environment was incomplete and did not contain PyTorch.

Installed packages before deletion:

```text
packaging==26.2
```
