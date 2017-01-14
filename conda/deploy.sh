#!/bin/bash -x
set -e
export PATH="$HOME/miniconda/bin:$PATH"
anaconda -t $ANACONDA_TOKEN upload --force $HOME/miniconda/conda-bld/*/smcpp-*.tar.bz2

# Next, build 
constructor conda
