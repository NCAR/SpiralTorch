from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# run using 
# python setup.py install

setup(
    name='fista_subproblem',
    ext_modules=[
        CUDAExtension('fista_cuf', [
            'fista_cuf.cpp',
            'fista_cuf_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })