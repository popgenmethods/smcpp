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

# see http://openmp.org/wp/openmp-compilers/
omp_test = \
r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

def check_for_openmp():
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    open(filename, 'w').write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call([os.environ.get("CC", 'cc'), '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)
    return result == 0

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        for w in ["-Wstrict-prototypes"]:
            cfg_vars[key] = value.replace(w, "")

cpps = [f for f in glob.glob("src/*.cpp") if 
        not os.path.basename(f).startswith("_") 
        and not os.path.basename(f).startswith("test")]

extra_compile_args=["-O2", "-std=c++11", "-Wno-deprecated-declarations", "-DNO_CHECK_NAN"]
# extra_compile_args=["-O0", "-g", "-std=c++11", "-Wno-deprecated-declarations"]
extra_link_args=[]

libraries = ['mpfr', 'gmp', 'gmpxx', 'gsl', 'gslcblas']
if check_for_openmp():
    extra_compile_args.append('-fopenmp')
    extra_link_args.append("-fopenmp")
else:
    warnings.warn("OpenMP compiler support not detected. Compiling SMC++ with OpenMP support is "
                  "*highly recommended* for performance reasons.")

if os.environ.get("SMCPP_STATIC", False):  # static link
    libraries = [':lib%s.a' % lib for lib in libraries]
    extra_link_args.append("-L%s" % os.environ['SMCPP_STATIC'])
    extra_link_args.append("-static-libstdc++")
    extra_link_args.append("-static-libgcc")

def lazy_extensions():
    # Lazy evaluation allows us to use setup_requires without have to import at
    # top level
    from Cython.Build import cythonize
    import pkgconfig
    import numpy as np
    include_dirs = ['/usr/local/include']
    for dep in ['gsl', 'mpfr']:
        include_dirs += [path.strip() for path in pkgconfig.cflags(dep).split("-I") if path.strip()]
    extensions = [
            Extension(
                "smcpp._smcpp",
                sources=["smcpp/_smcpp.pyx"] + cpps,
                language="c++",
                include_dirs=[np.get_include(), "include", "include/eigen3"] + include_dirs,
                library_dirs=['/usr/local/lib'],
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
                )]
    if False:
        extensions.append(  # This depends on boost and is only used for testing purposes
                Extension(
                    "smcpp._newick",
                    sources=["smcpp/_newick.pyx"],
                    include_dirs=["include"],
                    language="c++",
                    extra_compile_args=["-O2", "-std=c++11", "-Wfatal-errors", "-Wno-unused-variable", "-Wno-unused-function"]
                    )
                )
    return cythonize(extensions)

dist.Distribution({'setup_requires': ['numpy', 'pkgconfig', 'cython']})

setup(name='smcpp',
        description='SMC++',
        author='Jonathan Terhorst, Jack Kamm, Yun S. Song',
        author_email='terhorst@stat.berkeley.edu',
        url='https://github.com/popgenmethods/smc++',
        ext_modules=lazy_extensions(), # cythonize(extensions),
        packages=find_packages(),
        setup_requires=['pytest-runner', 'pkgconfig', 'cython', 'setuptools_scm'],
        use_scm_version={'write_to': "smcpp/version.py"},
        tests_require=['pytest'],
        install_requires=[
            "seaborn",
            "progressbar2",
            "setuptools_scm",
            "appdirs",
            "backports.shutil_which",
            "backports.shutil_get_terminal_size",
            "futures",
            "wrapt>=1.10",
            "setuptools>=19.6",
            "ad>=1.2.2",
            "cython>=0.23",
            "scipy>=0.16",
            "numpy>=1.9",
            "matplotlib>=1.5",
            "pysam>=0.9",
            "pandas",
            "future"],
        entry_points = {
            'console_scripts': ['smc++ = smcpp.frontend.console:main'],
            }
    )


