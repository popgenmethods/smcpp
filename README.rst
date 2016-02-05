============
Requirements
============

SMC++ requires the following external dependencies to build:

  - A compiler which supports C++11 (e.g. GCC 4.8 or later).
  - gmp <https://gmplib.org/> for some rational field computations.
  - gsl, the GNU Scientific Library.
  - Eigen <http://eigen.tuxfamily.org/>, a C++ linear algebra library.

If your compiler supports it (most do), SMC++ will run in multi-threaded
mode using OpenMP.

==================
Build Instructions
==================
The easiest way is to check out the git repo and build with distutils
using make.::
    $ git clone https://github.com/terhorst/psmcpp.git
    $ cd psmcpp
    $ git checkout current
    $ make

(Eventually)::
    $ pip install --user git+https://github.com/terhorst/psmcpp.git@current

=====
Usage
=====
Model estimation is accomplished using `scripts/em.py`. The first
required argument is a text-based configuration file. Following that,
one or more data files must be passed in. Each data file is assumed
to represent an independent sequence of observed bases (i.e., a
chromosome). An example configuration file and dataset may be found
in the `example/` directory.

Data Format
-----------
The data files should be ASCII text and can optionally be gzipped. The
format of each line of the data file is as follows:::

    <span>  <distinguished> <undistinguished>   <# undistinguished>

The first column gives the number of contiguous bases at which this
observation occurred. Hence, it will geneally be 1 for SNPs and >1 for
a stretch of nonsegregating sites. The second column gives the genotype
(0, 1, or 2) of the in the distinguished individual. If the genotype of
the distinguished individual is not known, this should be set to -1.
The final columns give the total number of derived alleles found in the
remainder of the (undistinguished) sample, as well as the *haploid*
sample size (number of non-missing observations) in that sample. 

For example, consider the following set of genotypes at a set of 10
contiguous bases on three diploid individuals::

    dist.   ..1..N...2
            .....N...1
            2N........

The distinguished individual is row one. A `.` indicates that the
individual is homozygous for the ancestral allele, while an integer
indicates that that individual possesses `(1,2)` copies of the derived
allele. Finally, an `N` indicates a missing genotype at that position.

The SMC++ format for this input file is::

    1   0   2   4
    1   0   0   2
    1   1   0   4
    2   0   0   0
    1   -1  0   2
    3   0   0   0
    1   2   1   4

Output
------
Upon completion, SMC++ will output a tab-delimeted table containing
the estimation results. The three columns `a`, `b`, and `s` define a
piecewise population model such that the estimated effective population
size `s` generations in the past, `eta(s)`, is:::

    eta(s) = a[i] * exp(log(b[i]/a[i])/(s[i] - s[i-1]) * (s - s[i-1])), s[i-1] <= s < s[i],

where we define `s[0] = 0` by convention. 
