import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext as _build_ext
from subprocess import call
import os.path
import glob

ignore = ["piecewise_polynomial", "spline_rate_function", "test_gradient", "loglik", "test_sad", "test_pe"]
cpps = [f for f in glob.glob("src/*.cpp") if 
        not os.path.basename(f).startswith("_") 
        and not os.path.basename(f).startswith("test") 
        and not any(f.endswith(x + ".cpp") for x in ignore)]

extensions = [
        Extension(
            "_pypsmcpp",
            sources=["src/_pypsmcpp.pyx"] + cpps,
            language="c++",
            include_dirs=["src", "/usr/include/eigen3", "/usr/local/include/eigen3", np.get_include(), "/export/home/terhorst/opt/include/"],
            # extra_compile_args=["-O0", "-ggdb3", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function", "-D_GLIBCXX_DEBUG"],
            extra_compile_args=["-O2", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function", "-fopenmp"],
            libraries=['stdc++', 'gmp', 'gmpxx', 'gsl', 'gslcblas'],
            library_dirs=["/export/home/terhorst/opt/lib"],
            extra_link_args=['-fopenmp']
            ),
        Extension(
            "_newick",
            # sources=["src/_pypsmcpp.pyx", "src/conditioned_sfs.cpp", "src/hmm.cpp"],
            sources=["src/_newick.pyx"],
            language="c++",
            extra_compile_args=["-O3", "-DNDEBUG", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function"],
            libraries=['stdc++'],
            ),
        ]

setup(name='psmc++',
        version='0.01',
        description='PSMC++',
        author='Jonathan Terhorst, Jack Kamm, Yun S. Song',
        author_email='terhorst@stat.berkeley.edu',
        url='https://github.com/terhorst/psmc++',
        ext_modules=cythonize(extensions),
        install_requires=[
            "cython>=0.23",
            "scipy>=0.16",
            "numpy>=1.9"]
    )
