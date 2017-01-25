#!/bin/bash -ex
export PATH=$PREFIX/lib/ccache/bin:$PATH
echo $(which gcc)
python setup.py install --single-version-externally-managed --record=/dev/null
