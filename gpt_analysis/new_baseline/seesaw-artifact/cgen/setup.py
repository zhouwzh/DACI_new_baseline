import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='cgen',
    ext_modules=[
        CUDAExtension(
            'cgen.page',
            ['csrc/page.cpp', 'csrc/page_kernel.cu'],
        )
    ],
    # packages=find_packages(include=["cgen"], exclude=("csrc", "scripts")),
    cmdclass={
        'build_ext': BuildExtension
    }
)
