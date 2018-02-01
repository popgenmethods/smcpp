name: smcpp
version: {version}
channels:
  - http://repo.continuum.io/pkgs/main/
  - http://repo.continuum.io/pkgs/free/
  - http://conda.anaconda.org/bioconda
  - http://conda.anaconda.org/terhorst
specs:
  - smcpp {version}
  - python 3.6.*
  - conda
post_install: post_install.sh
license_file: ../LICENSE
