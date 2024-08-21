# SpiralTorch
Library using PyTorch to implement Spiral-TAP, Sparsa and FISTA for total variation denoising

# Python Environment
Installing packages using conda can be accomplished with an anaconda installation as follows

Create a new python environment
Create the python environment (in this exampled named `ptv-casper-cuda`) with python version aligning with the pytorch version selected above:
```
conda create -n ptv-casper-cuda python=3.12
conda activate ptv-casper-cuda
```
Install base packages
```
conda install matplotlib numpy pyyaml scipy xarray ipykernel netcdf4 jupyter
conda install scikit-learn -c conda-forge
conda install ninja  # only needed for using the custom cuda kernel
```
If you are only installing `PyTorch` for CPU, the installation script
```
conda install pytorch -c pytorch  # this was tested with 2.3.0, but there is no reason to think other versions won't work
```

If you are planning to use a GPU, you will need to ensure that the CUDA compiler version is aligned with the install request.  E.g.
```
conda install pytorch=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```



# Install on Casper
Part of the challenge in making an environment that leverages the GPUs is ensuring that there is alignment between the CUDA compiler versions and the CUDA compiler used by PyTorch.  Here are the steps to install a GPU enabled environment through the example of Casper.

The NCAR conda module is way faster than the miniconda one in my home directory
```
module load conda
```

To list the available versions of cuda on the NCAR HPC systems
```
module spider cuda
```

Search versions of pytorch available for one that is compatible with the version of cuda and python available.
```
conda search pytorch -c pytorch
```
an example of the package we might want to install is
```
pytorch                        2.3.0 py3.12_cuda11.8_cudnn8.7.0_0  pytorch
```
which tells us we need cuda 11.8 and we should install pytoch 2.3.0

Create the python environment (in this exampled named `ptv-casper-cuda`) with python version aligning with the pytorch version selected above:
```
conda create -n ptv-casper-cuda python=3.12
conda activate ptv-casper-cuda
```

Import the cuda module, make sure the version corresponds to the expected install version for pytorch (from the search above)
```
module swap intel gcc
module load cuda/11.8
export CONDA_OVERRIDE_CUDA="11.8"
```
Install libraries
```
conda install matplotlib numpy pyyaml scipy xarray ipykernel netcdf4
conda install scikit-learn -c conda-forge
conda install ninja
conda install lmod  # this is for setting the environment on JupyterHub
```
Install pytorch libraries with cuda (for `cuda/11.8`)
```
conda install pytorch=2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
(for `cuda/12.1`)
```
conda install pytorch=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Fixing compiler checks in PyTorch
PyTorch uses `-v` arguments to check compatability of compilers which will fail on the NCAR HPC systems (this argument will try to compile if it can, fail, and cause the program to fail).  We have to edit the PyTorch source code and change a few of these arguments to `--version`.

I usually have to edit
```
conda-envs/ptv-casper-cuda/lib/python3.12/site-packages/torch/utils/cpp_extension.py
```

Note that the `-v` flag does not always mean "version".  The `ninja` flag `-v` is equivalent to `--verbose`.  Don't change this to `--version` in the `cpp_extension.py` or it breaks the compile step!

# Compiling CUDA for FISTA
There are two options for the FISTA function used in the gradient calculation.  `jit-fista` uses PyTorch's precompiler and is built on the PyTorch types and architecture.  

A faster version of FISTA (`fista_cuf` or `cuda-fista`) is a compiled cuda kernel.  While this approach is faster, it needs to be compiled on the system.  These are the steps to do this on the NSF HPC systems.

The method for creating a custom cuda kernel in this repo follows this tutorial

https://pytorch.org/tutorials/advanced/cpp_extension.html

However this method is apparently being replaced in PyTorch.

Starting in PyTorch 2.4 the method for creating custom cuda operators changes

https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#testing-an-operator

This will require modifications to how the package is setup but the new tutorial is a bit hard to follow.

.......
Original command but I don't think we need torchvision or torchaudio
```
conda install pytorch=2.3.0 pytorch-cuda=11.8 torchvision=0.15.0 torchaudio=2.0.0  -c pytorch -c nvidia

```

.....

```
:module swap intel gcc

Due to MODULEPATH changes, the following have been reloaded:
  1) hdf5/1.12.2     2) ncarcompilers/1.0.0     3) netcdf/4.9.2     4) openmpi/4.1.6
:echo $CXX
g++

:swap cuda cuda/11.8

The following have been reloaded with a version change:
  1) cuda/12.2.1 => cuda/11.8.0

conda activate ptv-torch-cuda
```
navigate to `python/SpiralTorch/cuda`

```
python setup.py install
```

# Running CUDA module on Casper Update Aug 2024
successful compile

module settings
```
module load conda
conda activate ptv-casper-cuda12
```

```
module swap intel gcc
module load cuda/12
export CONDA_OVERRIDE_CUDA="12.2"
module load gcc/12
```