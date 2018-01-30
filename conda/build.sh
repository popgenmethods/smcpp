#!/bin/bash -ex
export CFLAGS="-I$PREFIX/include $CFLAGS"
export CXXFLAGS="-I$PREFIX/include -Wno-int-in-bool-context $CXXFLAGS"
set
python setup.py install --single-version-externally-managed --record=/dev/null
