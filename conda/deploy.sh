#!/bin/bash
export PATH="$HOME/miniconda/bin:$PATH"
conda convert --platform all $HOME/miniconda/conda-bld/**/smcpp-*.tar.bz2 --output-dir conda-bld/
anaconda -t $ANACONDA_TOKEN upload --force conda-bld/**/smcpp-*.tar.bz2

# Next, build 
constructor conda
