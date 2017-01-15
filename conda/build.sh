#!/bin/bash

export CC=gcc-5 CXX=g++-5
echo $(which gcc-5)
echo $(which g++-5)
$PYTHON setup.py install --single-version-externally-managed --record=/dev/null

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
