#!/bin/bash

BINARY=dist/smcpp-$TRAVIS_OS_NAME

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    brew update
    brew install mpfr gmp gsl homebrew/versions/gcc5
    OS=MacOSX
else
    sudo apt-get -qq update
    sudo apt-get install -y libmpc-dev libmpfr-dev libgmp-dev libgsl0-dev 
	OS=Linux
fi

wget http://repo.continuum.io/miniconda/Miniconda3-latest-$OS-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n test-environment numpy scipy matplotlib pandas dateutil Cython pysam curl openblas libpython
source activate test-environment
pip install --upgrade pip
pip install --upgrade setuptools
pip install pyinstaller==3.1.1 packaging
pip install -r requirements.txt
CC=gcc-5 CXX=g++-5 python setup.py develop
pyinstaller --clean -F --exclude PyQt5 --exclude PyQt4 --exclude pyside scripts/smc++
dist/smc++ estimate -h
dist/smc++ vcf2smc -h
mv dist/smc++ $BINARY
