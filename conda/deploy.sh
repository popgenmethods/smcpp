#!/bin/bash -x
set -e
export PATH="$HOME/miniconda/bin:$PATH"
PKGS=$(conda build -c terhorst -c conda-forge -c bioconda conda)
anaconda -t $ANACONDA_TOKEN upload --force $PKGS

# Next, build 
constructor conda
