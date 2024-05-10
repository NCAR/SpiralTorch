from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# run using 
# python setup.py install

setup(
    name='st_fista_subproblem',
    ext_modules=[
        CUDAExtension('st_fista_cuf', [
            'st_fista_cuf.cpp',
            'st_fista_cuf_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })