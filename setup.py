from __future__ import print_function
from setuptools import setup, Extension, find_packages, dist
import os
import os.path
import glob
import sys
import tempfile
import subprocess
import shutil
import warnings
import numpy as np

if True:
    extra_compile_args = [
        "-O2",
        "-std=c++11",
        "-Wno-deprecated-declarations",
        "-Wno-int-in-bool-context",
        "-DNO_CHECK_NAN",
        "-fopenmp",
        "-DEIGEN_DONT_PARALLELIZE",
    ]
else:
    extra_compile_args = [
        "-O0",
        "-g",
        "-std=c++11",
        "-Wno-deprecated-declarations",
        "-Wno-int-in-bool-context",
        "-fopenmp",
        "-D_GLIBCXX_DEBUG",
        "-Wfatal-errors",
        "-DEIGEN_DONT_PARALLELIZE",
    ]

extra_link_args = ["-fopenmp"]
libraries = ["mpfr", "gmp", "gmpxx", "gsl", "gslcblas"]
cpps = [
    f
    for f in glob.glob("src/*.cpp")
    if not os.path.basename(f).startswith("_")
    and not os.path.basename(f).startswith("test")
]

extensions = [
    Extension(
        "smcpp._smcpp",
        sources=["smcpp/_smcpp.cpp"] + cpps,
        language="c++",
        include_dirs=[np.get_include(), "include", "include/eigen3"],
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "smcpp._estimation_tools",
        sources=["smcpp/_estimation_tools.c"],
        include_dirs=[np.get_include()],
        libraries=libraries,
    ),
]

setup(
    name="smcpp",
    description="SMC++",
    author="Jonathan Terhorst, Jack Kamm, Yun S. Song",
    author_email="jonth@umich.edu",
    url="https://github.com/popgenmethods/smc++",
    ext_modules=extensions,
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "appdirs",
        "scipy>=1.8",
        "numpy>=1.18",
        "matplotlib>=3",
        "pysam>=0.18",
        "pandas >=1.4",
        "scikit-learn>=1",
    ],
    entry_points={"console_scripts": ["smc++ = smcpp.frontend.console:main"],},
)
