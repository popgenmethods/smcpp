#!/bin/bash -ex
export CFLAGS="-I$PREFIX/include -Wno-int-in-bool-context $CFLAGS"
export CXXFLAGS="-I$PREFIX/include -Wno-int-in-bool-context $CXXFLAGS"
$PYTHON setup.py install --single-version-externally-managed --record=/dev/null
