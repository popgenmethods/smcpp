#!/bin/bash -x
set -e
export PATH="$HOME/miniconda/bin:$PATH"
PKGS=$(conda build -c terhorst -c conda-forge -c bioconda conda --output)
PKGS=$(cat .packages)
anaconda -t $ANACONDA_TOKEN upload --force $PKGS

# Next, pull uploaded pkg and create a binary installed
constructor conda
