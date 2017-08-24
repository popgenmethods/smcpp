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
from Cython.Build import cythonize
import numpy as np

if True:
    extra_compile_args = ["-O2", "-std=c++11", "-Wno-deprecated-declarations", "-DNO_CHECK_NAN", 
                          '-fopenmp']
else:
    extra_compile_args = ["-O0", "-g", "-std=c++11", "-Wno-deprecated-declarations",
                          "-D_GLIBCXX_DEBUG", "-Wfatal-errors"]

extra_link_args=['-fopenmp']
libraries = ['mpfr', 'gmp', 'gmpxx', 'gsl', 'gslcblas']
cpps = [f for f in glob.glob("src/*.cpp") if
        not os.path.basename(f).startswith("_")
        and not os.path.basename(f).startswith("test")]

extensions = [
        Extension(
            "smcpp._smcpp",
            sources=["smcpp/_smcpp.pyx"] + cpps,
            language="c++",
            include_dirs=[np.get_include(), "include", "include/eigen3"],
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
            )]

setup(name='smcpp',
        description='SMC++',
        author='Jonathan Terhorst, Jack Kamm, Yun S. Song',
        author_email='terhorst@stat.berkeley.edu',
        url='https://github.com/popgenmethods/smc++',
        ext_modules=cythonize(extensions),
        packages=find_packages(),
        setup_requires=['cython>=0.25', 'setuptools_scm'],
        use_scm_version={'write_to': "smcpp/version.py"},
        install_requires=[
            "tqdm",
            "setuptools_scm",
            "appdirs",
            "wrapt>=1.10",
            "ad>=1.3.2",
            "scipy>=0.16",
            "numpy>=1.9",
            "matplotlib>=1.5",
            "pysam>=0.9",
            "attrs>=17.2.0",
            "pandas"],
        entry_points = {
            'console_scripts': ['smc++ = smcpp.frontend.console:main'],
            }
    )
