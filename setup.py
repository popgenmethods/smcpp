from __future__ import print_function
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os.path
import glob

cpps = [f for f in glob.glob("src/*.cpp") if 
        not os.path.basename(f).startswith("_") 
        and not os.path.basename(f).startswith("test") ]

extensions = [
        Extension(
            "smcpp._smcpp",
            sources=["src/_smcpp.pyx"] + cpps,
            language="c++",
            include_dirs=["src", "/usr/include/eigen3", "/usr/local/include/eigen3", np.get_include(), "/export/home/terhorst/opt/lib"],
            # extra_compile_args=["-O0", "-ggdb3", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function", "-D_GLIBCXX_DEBUG"],
            extra_compile_args=["-O2", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function", "-fopenmp"],
            libraries=['gmp', 'gmpxx', 'gsl', 'gslcblas'],
            extra_link_args=['-fopenmp']
            ),
        Extension(
            "smcpp._newick",
            # sources=["src/_pypsmcpp.pyx", "src/conditioned_sfs.cpp", "src/hmm.cpp"],
            sources=["src/_newick.pyx"],
            language="c++",
            extra_compile_args=["-O2", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function"],
            ),
        ]


print(find_packages())
setup(name='smcpp',
        version='0.1.0',
        description='SMC++',
        author='Jonathan Terhorst, Jack Kamm, Yun S. Song',
        author_email='terhorst@stat.berkeley.edu',
        url='https://github.com/terhorst/smc++',
        ext_modules=cythonize(extensions),
        packages=find_packages(),
        install_requires=[
            "cython>=0.23",
            "scipy>=0.16",
            "numpy>=1.9",
            "matplotlib>=1.5",
            "future"],
        extras_require = {'gui': ["Gooey>=0.9"]},
        entry_points = {
            'console_scripts': ['smc++ = smcpp.frontend.console:main'],
            'gui_scripts': ['smc++-gui = smcpp.frontend.gui:main [gui]']
            }
    )
