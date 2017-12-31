#!/bin/bash -ex
CFLAGS="-I$PREFIX/include -Wno-int-in-bool-context -mtune=generic $CFLAGS"
CXXFLAGS="-Wno-int-in-bool-context -I$PREFIX/include -mtune=generic $CXXFLAGS"
LDFLAGS="-L$PREFIX/lib $LDFLAGS"
PATH="$PREFIX/bin:$PATH"
LDSHARED=$(python -c "import sys;from distutils import sysconfig;lds = sysconfig.get_config_var('LDSHARED').split(' ');print(' '.join(['$GCC'] + lds[1:]))")
export CFLAGS CXXFLAGS LDFLAGS PATH LDSHARED
$PYTHON setup.py install --single-version-externally-managed --record=/dev/null
