name: smcpp
version: {version}
channels:
  - http://repo.continuum.io/pkgs/free/
  - http://conda.anaconda.org/bioconda
  - http://conda.anaconda.org/terhorst
  - http://conda.anaconda.org/conda-forge
  - http://conda.anaconda.org/salford_systems
specs:
  - isl  # necessary to fix a bug
  - gcc  # ditto
  - smcpp
license_file: ../LICENSE
