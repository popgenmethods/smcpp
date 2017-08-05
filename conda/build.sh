#!/bin/bash -ex
export CFLAGS="-mtune=generic" CXXFLAGS="-mtune=generic"
$PYTHON setup.py install --single-version-externally-managed --record=/dev/null
