from __future__ import print_function
from setuptools import setup, Extension, find_packages, dist
import os
import os.path
import glob
import sys

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

cpps = [f for f in glob.glob("src/*.cpp") if 
        not os.path.basename(f).startswith("_") 
        and not os.path.basename(f).startswith("test")]

def lazy_extensions():
    # Lazy evaluation allows us to use setup_requires without have to import at
    # top level
    from Cython.Build import cythonize
    import numpy as np
    import pkgconfig
    include_dirs = []
    for dep in ['gsl']:
        include_dirs += [path.strip() for path in pkgconfig.cflags(dep).split("-I") if path.strip()]
    extensions = [
            Extension(
                "smcpp._smcpp",
                sources=["smcpp/_smcpp.pyx"] + cpps,
                language="c++",
                include_dirs=["include", "include/eigen3", np.get_include()] + include_dirs,
                # extra_compile_args=["-O0", "-ggdb3", "-std=c++11", "-Wno-unused-variable",
                #    "-Wno-unused-function", "-D_GLIBCXX_DEBUG", '-fopenmp'],
                extra_compile_args=["-O2", "-g", "-std=c++11", "-Wno-deprecated-declarations", "-fopenmp"],
                libraries=['gmp', 'gmpxx', 'gsl', 'gslcblas', 'mpfr'],
                extra_link_args=['-rdynamic', '-fopenmp']
                )]
    if True:
        extensions.append(## This depends on boost and is only used for testing purposes
                Extension(
                    "smcpp._newick",
                    # sources=["src/_pypsmcpp.pyx", "src/conditioned_sfs.cpp", "src/hmm.cpp"],
                    sources=["smcpp/_newick.pyx"],
                    include_dirs=["include"],
                    language="c++",
                    extra_compile_args=["-O2", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function"]
                    )
                )
    return cythonize(extensions)

## Create a dummy distro in order to get setup_requires without
## having to have already installed these modules
## numpy auto install doesn't work.
try:
    import numpy
except ImportError:
    sys.exit("Setup requires numpy order to proceed. "
             "Please install it and and re-run.")
dist.Distribution({'setup_requires': ['numpy', 'pkgconfig', 'cython']})



setup(name='smcpp',
        version=os.environ["GIT_VERSION"],
        description='SMC++',
        author='Jonathan Terhorst, Jack Kamm, Yun S. Song',
        author_email='terhorst@stat.berkeley.edu',
        url='https://github.com/terhorst/smc++',
        ext_modules=lazy_extensions(), # cythonize(extensions),
        packages=find_packages(),
	setup_requires=['pytest-runner', 'numpy', 'pkgconfig', 'cython'],
        tests_require=['pytest'],
        install_requires=[
            "gmpy2", # This is not used in the Python code, but 
                     # enforces the libmpfr and libgmp dependencies.
            "appdirs",
            "backports.shutil_which",
            "backports.shutil_get_terminal_size",
            "wrapt>=1.10",
            "setuptools>=19.6",
            "jsonpickle>=0.9.2",
            "ad>=1.2.2",
            "cython>=0.23",
            "scipy>=0.16",
            "numpy>=1.9",
            "matplotlib>=1.5",
            "pysam>=0.9",
            "future"],
        extras_require = {'gui': ["Gooey>=0.9"]},
        entry_points = {
            'console_scripts': ['smc++ = smcpp.frontend.console:main'],
            'gui_scripts': ['smc++-gui = smcpp.frontend.gui:main [gui]']
            }
    )
