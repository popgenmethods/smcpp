SMC++ is a program for estimating the size history of a population from
whole genome sequence data.

=================
Quick Start Guide
=================
Follow these steps to get up and running as quickly as possible:

0. Make sure you have the requirements installed. (See section
   "Requirements" below.)
1. Install SMC++ via `pip` (this will change once we upload to PyPI)::

     $ pip install git+https://github.com/terhorst/pmscpp.git

   Depending on your platform, `pip` will either download a pre-compiled
   binary, or compile SMC++ from scratch.
2. Convert your VCF file to the `smc++` format::

     $ smc++ vcf2smc example.vcf chr1 data/example.chr1.smc.gz

   Repeat as many times an needed for different chromosomes.
3. Fit the model::

     $ smc++ estimate results/ 1.2e-8 3e-9 data/example.chr*.smc.gz

   Depending on sample size and your machine, the fitting procedure
   should take between a few minutes and a few hours. The fitted model
   will be stored in JSON format in `results/model.final.json`. For details
   on the format, see below.
4. Visualize the results::

     $ smc++ plot_model results/model.final.json results/fit.pdf

There is also a graphical user interface to each of these commands, which
is accessed by running::

     $ smc++-gui

(Currently, this only works for Python 2.x.)


============
Requirements
============
SMC++ requires the following external dependencies to build:

  - Python 2.7 or greater. SMC++ is compatible with Python 3, but only
    in console mode.
  - A compiler which supports C++11 (e.g. GCC 4.8 or later) *and*
    OpenMP. Note that versions of Clang shipping with Mac OS X do not
    currently support OpenMP. For this reason it is recommended that you
    use gcc instead.
  - gmp <https://gmplib.org/> for some rational field computations.
  - mpfr <http://www.mpfr.org/> for some extended precision calculations.
  - gsl, the GNU Scientific Library.

On Ubuntu (or Debian) Linux, the library requirements may be installed
using the commmand::

    $ sudo apt-get install -y libgmp-dev libmpfr-dev libgsl0-dev

On OS X, the easiest way to install them is using Homebrew <http://brew.sh/>::

    $ brew install mpfr gmp gsl gcc

Experimental pre-built binaries are available for Unix and Mac OS X
systems. They will download automatically using `pip` (see above)
if available for your system. Note that you will still need to have
`libgmp`, `libgsl` and `libmpfr` accessible on your system in order 
to run SMC++.

=================
Input Data Format
=================
In case the provided tool `smc++ vcf2smc` is insufficient for your
purposes, here is a description of the input format used by SMC++.
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
            2N....+...

The distinguished individual is row one. A `.` indicates that the
individual is homozygous for the ancestral allele, while an integer
indicates that that individual possesses `(1,2)` copies of the derived
allele. An `N` indicates a missing genotype at that position. Finally,
the `+` in column seven indicates that individual three possessed the
dominant allele on one chromosome, and had a missing observation on the
other chromosome (this would be indicated as `0/.` in a VCF).

The SMC++ format for this input file is::

    1   0   2   4
    1   0   0   2
    1   1   0   4
    2   0   0   4
    1   -1  0   2
    1   0   0   3
    2   0   0   0
    1   2   1   4

=============
Output Format
=============
Upon completion, SMC++ will output a tab-delimeted table containing
the estimation results. The three columns `a`, `b`, and `s` define a
piecewise population model such that the estimated effective population
size `s` generations in the past, `eta(s)`, is:::

    eta(s) = a[i] * exp(log(b[i]/a[i])/(s[i] - s[i-1]) * (s - s[i-1])), s[i-1] <= s < s[i],

where we define `s[0] = 0` by convention. Note that the population      
sizes `a` and `b` are the *diploid* effective population size at each   
corresponding time interval.                                            
