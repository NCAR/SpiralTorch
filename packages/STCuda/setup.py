import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "STCuda"

import importlib.util
spec = importlib.util.spec_from_file_location("version", library_name + "/version.py")
ver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ver)

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"

    # Casper has both V100 (sm_70) and A100 (sm_80) GPU nodes. Default to a multi-arch
    # ("fat") build so a single compiled .so runs natively on either card. Without this,
    # CUDAExtension auto-detects only the *build* node's GPU, so the kernel carries one arch
    # and fails on the other (CUDA error 222 "the provided PTX was compiled with an
    # unsupported toolchain", when the driver must JIT the embedded PTX). setdefault keeps it
    # overridable via `TORCH_CUDA_ARCH_LIST=...`. As a bonus this also avoids the login-node
    # build crash (arch auto-detect IndexError when no GPU is present).
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;8.0")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "-lineinfo",
            "-rdc=true",
            "-use_fast_math"
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))
    sources += cuda_sources

    ext_modules = [
        CUDAExtension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            dlink=True,
            dlink_libraries=["cudadevrt"],
            libraries=["cudart", "cudadevrt"],
            py_limited_api=py_limited_api,
        )
    ]
    return ext_modules

setup(
    name=library_name,
    version=ver.__version__,
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="FISTA subproblem CUDA extension",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
