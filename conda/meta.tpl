package:
  name: smcpp
  version: {version}

source:
  path: ../
  # git_rev: v{version}
  # git_url: https://github.com/terhorst/psmcpp.git

requirements:
  build:
    - python 3.5.2
    - ccache 3.3.2
    - gcc 4.8.5
    - libgcc 4.8.5
    - gmp 6.1.0
    - gsl 2.2.1
    - mpc 1.0.3
    - mpfr 3.1.5
    - cython 0.25.2
    - numpy 1.10.4
    - setuptools_scm 1.15.0
  run:
    - conda
    - nomkl
    - gnuplot 5.0.5
    - python 3.5.2
    - libgcc 4.8.5
    - gmp 6.1.0
    - gsl 2.2.1
    - mpc 1.0.3
    - mpfr 3.1.5
    - numpy 1.11.3
    - setuptools_scm 1.15.0
    - pysam 0.9.1.4
    - matplotlib 2.0.0
    - pandas 0.19.2
    - python-dateutil 2.6.0
    - scipy 0.18.1
    - six 1.10.0
    - appdirs 1.4.0
    - tqdm 4.10.0
    - seaborn 0.7.1
    - wrapt 1.10.8
    - ad 1.3.2

about:
  home: https://github.com/popgenmethods/smcpp
  license: BSD
  license_file: LICENSE

test:
  imports:
    - smcpp
    - smcpp._smcpp
