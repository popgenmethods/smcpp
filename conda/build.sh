#!/bin/bash -ex
export CFLAGS="-I$PREFIX/include $CFLAGS"
export CXXFLAGS="-Wno-int-in-bool-context $CXXFLAGS"
python setup.py install --single-version-externally-managed --record=/dev/null
