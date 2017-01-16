#!/bin/bash -x
set -e
export PATH="$HOME/miniconda/bin:$PATH"
anaconda -t $ANACONDA_TOKEN upload --force $(conda build --output conda)

# Next, build 
constructor conda
