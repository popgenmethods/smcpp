#!/bin/bash

BINARY=dist/smcpp-$TRAVIS_OS_NAME

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    OS=MacOSX
else
	OS=Linux
fi

wget http://repo.continuum.io/miniconda/Miniconda3-latest-$OS-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda env create -f .conda.yml
source activate test-environment
CC=gcc CXX=g++ python setup.py develop
pip install git+https://github.com/pyinstaller/pyinstaller@483c819
pyinstaller --clean -s -F --exclude PyQt5 --exclude PyQt4 --exclude pyside scripts/smc++
dist/smc++ vcf2smc example/example.vcf.gz example/example.smc.gz 1 msp:msp_0,msp_1,msp_2
dist/smc++ estimate --theta .00025 --em-iterations 1 example/example.smc.gz
mv dist/smc++ $BINARY
