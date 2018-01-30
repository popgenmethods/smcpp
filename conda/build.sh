#!/bin/bash -ex
export CFLAGS="-I$PREFIX/include $CFLAGS"
python setup.py install --single-version-externally-managed --record=/dev/null
