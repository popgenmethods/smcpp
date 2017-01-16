#!/bin/bash -ex
mkdir -p compilers
ln -s `which ccache` compilers/gcc
ln -s `which ccache` compilers/g++
PATH=$PWD/compilers:$PATH python setup.py install --single-version-externally-managed --record=/dev/null
