# SpiralTorch
Library using PyTorch to implement Spiral-TAP, Sparsa and FISTA for total variation denoising.  This is a method for calculating gradients to solve Total Variation (TV) regularization problems.

The concept is that we wish to obtain a state variable $x$ from a physical model $f(x)$ relating the state to the noisy observations $y$.  Maximum likelihood estimation is employed to compare the parameterization of $f(x)$ to the noise observations so that estimate is

$\tilde{x} = \argmin L[y,f(x)] + \lambda ||x||_{TV}$

where $L(y,\alpha)$ is the statistical model for the noisy observations given parameterization $\alpha$, $||x||_{TV}$ is the total variation of $x$ and $\lambda$ is a real scalar that sets the total variation regularization.

# Examples using SpiralTorch

Examples of single variable and multi variable estimation can be found in
```
notebooks/casper/RunSpiral.ipynb
```
Also an example of backscatter and depolarization estimation is in
```
notebooks/casper/Spiral_MultiVariable_Estimation.ipynb
```

These notebooks are self contained within this repository (unlike some other examples that use actual MPD data).

# Python Environment
Installing packages using conda can be accomplished with an anaconda installation as follows

Create a new python environment
Create the python environment (in this exampled named `ptv-casper-cuda12`) with python version aligning with the pytorch version selected above:
```
conda create -n ptv-casper-cuda12 python=3.12
conda activate ptv-casper-cuda12
```
Install base packages
```
conda install matplotlib numpy pyyaml scipy xarray ipykernel netcdf4 jupyter
conda install scikit-learn -c conda-forge
conda install ninja  # only needed if using the custom cuda kernel for speedup
```
If you are only installing `PyTorch` for CPU, use the installation script
```
conda install pytorch -c pytorch  # this was tested with 2.3.0, but there is no reason to think other versions won't work
```

If you are planning to use a GPU, you will need to ensure that the CUDA compiler version is aligned with the install request.  E.g.
```
conda install pytorch=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
It is also important that you make sure your c++ compiler is compatable with the cuda compiler.  In the case of the available compilers on the Casper cluster, `nvcc 11` is not compatable with any `gcc > 11`.  Since Casper does not have any modules of `gcc <= 11` available, we have to use `nvcc 12` along with the `pytorch-cuda=12.1`.


# Install on Casper
Part of the challenge in building an environment that leverages the GPUs is ensuring that there is alignment between the CUDA compiler versions and the CUDA compiler used by PyTorch and the C++ compiler.  Here are the steps to install a GPU enabled environment through the example of the Casper cluster with a GPU (V100) node.

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
pytorch                        2.3.1 py3.12_cuda12.1_cudnn8.9.2_0  pytorch
```
which tells us we need cuda 12.1 (really any 12.x version) and we should install pytoch 2.3.0

Create the python environment (in this exampled named `ptv-casper-cuda12`) with python version aligning with the pytorch version selected above:
```
conda create -n ptv-casper-cud12 python=3.12
conda activate ptv-casper-cuda12
```

Import the cuda module, make sure the version corresponds to the expected install version for pytorch (from the search above)
```
module swap intel gcc
module load gcc/12  # not generally necessary, but helps control the gcc compiler version
module load cuda/12
export CONDA_OVERRIDE_CUDA="12.2"
```
Install libraries
```
conda install matplotlib numpy pyyaml scipy xarray ipykernel netcdf4
conda install scikit-learn -c conda-forge
conda install ninja
conda install lmod  # this is for setting the environment on JupyterHub
```
Install pytorch libraries with cuda (for `cuda/12.2`)
```
conda install pytorch=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Fixing compiler checks in older versions of PyTorch
Some older versions of PyTorch use `-v` arguments to check compatability of compilers which will fail on the NCAR HPC systems (this argument will try to compile if it can, fail, and cause the program to fail).  If this failure occurs at the compiler check line, switching the `-v` flag to `--version` will likely solve the problem.  This is done by editing the PyTorch source python directly in
```
[EnvironmentPath]/conda-envs/ptv-casper-cuda12/lib/python3.12/site-packages/torch/utils/cpp_extension.py
```

Note that the `-v` flag does not always mean "version".  The `ninja` flag `-v` is equivalent to `--verbose`.  Don't change this to `--version` in the `cpp_extension.py` or it breaks the compile step!

Note that in PyTorch 2.3.0 the following `try/except` case has been added to `cpp_extension.py` so that this issue does not arise
```
try:
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
except Exception as e:
    try:
        version_string = subprocess.check_output([compiler, '--version'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception as e:
        return False
```

# Compiling CUDA for FISTA
There are two options for the FISTA function used in the gradient calculation.  `jit-fista` uses PyTorch's precompiler and is built on the PyTorch types and architecture.  

A faster version of FISTA (`st_fista_cuf` or `cuda-fista`) is a compiled cuda kernel.  While this approach is faster, it needs to be compiled on the system.  These are the steps to do this on the NSF HPC systems.


### Note on the cuda definitions
The method for creating a custom cuda kernel in this repo follows this tutorial

https://pytorch.org/tutorials/advanced/cpp_extension.html

However this method is apparently being replaced in PyTorch.

Starting in PyTorch 2.4 the method for creating custom cuda operators changes

https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#testing-an-operator

This will require modifications to how the package is setup but the new tutorial is a bit hard to follow.