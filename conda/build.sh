#!/bin/bash -ex
mkdir -p $PREFIX/.bin
for c in gcc g++; do ln -s $(which ccache) $PREFIX/.bin/$c; done
export PATH=$PREFIX/.bin:$PATH CCACHE_BASEDIR=$PREFIX
echo $(which gcc)
ccache -s
python setup.py install --single-version-externally-managed --record=/dev/null
ccache -s
rm -rf $PREFIX/.bin
