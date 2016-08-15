SMC++ is a program for estimating the size history of a population from
whole genome sequence data.

=================
Quick Start Guide
=================

  1. Install the software. See `Installation`_, below.

  2. Convert your VCF(s) to the SMC++ input format with `vcf2smc`_::

         $ smc++ vcf2smc my.data.vcf.gz chr1 data/example.chr1.smc.gz

     This command will parse data for the contig `chr1` across all
     samples in the VCF. You should run this once for each independent
     contig in your dataset, producing one SMC++ output file per contig.

  3. Fit the model using `estimate`_::

       $ smc++ estimate -o analysis/ data/example.chr*.smc.gz
       
     Depending on sample size and your machine, the fitting procedure
     should take between a few minutes and a few hours. The fitted model
     will be stored in JSON format in `analysis/model.final.json`. For
     details on the format, see below.

  4. Visualize the results using `plot`_::

       $ smc++ plot --labels species1 -- analysis/model.final.json results/fit.pdf

============
Installation
============

SMC++ assumes that the following libraries and executables are available
on your system:

    - Python 2.7 or greater. SMC++ is compatible with Python 3, but only
      in console mode.
    - The `The GNU Multiple Precision Arithmetic Library <https://gmplib.org/>`_.
    - The `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_.

Installing the Dependencies
===========================

To install these requirements on Ubuntu Linux, use::

    $ sudo apt-get install python libgsl0-dev libgmp-dev

The easiest way to install these requirements on a Mac is using 
`Homebrew <http://brew.sh/>`_::

    $ brew install gmp gsl 

Binary Installation
===================

Compilation Instructions
========================
If binaries are not available for your platform, you can compile SMC++
from scratch by following these steps:

  1. Install additional required software:

    - A compiler which supports C++11 (e.g. GCC 4.8 or later) *and*
      OpenMP. Note that versions of Clang shipping with Mac OS X do not
      currently support OpenMP. For this reason it is recommended that you
      use gcc instead.
    - `Eigen <http://eigen.tuxfamily.org/>`_, a C++ linear algebra library.


  2. Install SMC++ via ``pip`` (this will change once we upload to PyPI)::

       $ pip install git+https://github.com/terhorst/pmscpp.git

     Depending on your platform, ``pip`` will either download a pre-compiled
     binary, or compile SMC++ from scratch.

Virtual Environment
===================

SMC++ requires in a fair number of Python dependencies.
If you prefer to keep this separate from your main Python
installation, or do not have root access on your system, you
may wish to install SMC++ inside of a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. To do so,
first create and activiate the virtual environment::

    $ virtualenv -p python2.7 <desired location>
    $ source <desired location>/bin/active

Then, install SMC++ using ``pip`` as described above.



There is also a graphical user interface to each of these commands, which
is accessed by running::

       $ smc++-gui



==========================
Description of subcommands
==========================

SMC++ comprises several subcommands which are accessed using the syntax 
``$ smc++ <subcommand>``.

vcf2smc
=======
This subcommand converts (biallelic, diploid) VCF data to the format used by
SMC++. 

Required arguments
------------------

    1. An `indexed VCF file <http://www.htslib.org/doc/tabix.html>`_.
    2. An output file. Appending the `.gz` extension will cause the output
       to be compressed; the `estimate` command can read from both compressed
       and uncompressed data sources.
    3. A contig name. Each call to `vcf2smc` processes a single contig. 
       VCFs containing multiple contigs should be processed via multiple
       calls to `vcf2smc`.

Optional arguments
------------------
Following the three required arguments, the user may append sample
names corresponding to columns in the VCF file. *If no sample names are
supplied, all samples in the VCF are converted*. The user should ensure
that this is the desired behavior since e.g. the VCF may contain samples
from multiple distinct populations.

SMC++ relies crucially on the notion of a *distinguished individual*
(see paper for details on this terminology). The identity of the
distinguished individual is set using the ``-i`` option, which specifies
the (zero-based) position in the sample list of the distinguished
individual. The default is ``-i 0``.

By varying ``-i`` over the same VCF, the user can create distinct data
sets for estimation. This is useful for forming composite likelihoods.
For example, the following command will create three data sets from
contig ``chr1`` of ``myvcf.gz``, by varying the identity of the distinguished
individual and treating the remaining two samples as "undistinguished":

.. code-block:: bash

    for i in {0..2}; 
        do smc++ vcf2smc -i $i myvcf.gz out.$i.txt chr1 NA12877 NA12878 NA12890; 
    done

Manual conversion
-----------------
``vcf2smc`` targets a common use-case but may not be sufficient for all
users. Those wishing to implement their own custom conversion to the SMC
data format should see the `input data format`_ description below.

estimate
========

     Here, the ``--theta`` option specifies a known mutation rate of
     :math:`\mu=1.25 \times 10^{-8}`/bp/gen in units of the reference
     effective population size :math:`2 N_0`. (The reference population
     size may be adjusted using the ``--N0`` switch.) If :math:`\theta` is not
     known for your species, it will be estimated from data using Watterson's
     estimator.

plot
====


============
File Formats
============

Input Data Format
=================
The data files should be ASCII text and can optionally be gzipped. The
format of each line of the data file is as follows::

    <span> <d> <u1> <n1> [<u2> <n2>]

Explanation of each column:

  - ``span`` gives the number of contiguous bases at which this
    observation occurred. Hence, it will generally be ``1`` for SNPs and
    greater than one for a stretch of nonsegregating sites.
  - ``d`` Gives the genotype (``0``, ``1``, or ``2``) of the
    distinguished individual. If the genotype of the distinguished
    individual is not known, this should be set to ``-1``.
  - The next column ``u1`` is the total number of derived alleles found
    in the remainder of the (undistinguished) sample at the site(s).
  - The final column ``n1`` is the *haploid* sample size (number of
    non-missing observations) in the undistinguished portion of the
    sample.
  - If two populations are to be analyzed, ``u2`` and ``n2`` are also 
    specified for the second population.

For example, consider the following set of genotypes at a set of 10
contiguous bases on three diploid individuals in one population::

    dist.   ..1..N...2
            .....N...1
            2N....+...

The distinguished individual is row one. A ``.`` indicates that the
individual is homozygous for the ancestral allele, while an integer
indicates that that individual possesses ``(1,2)`` copies of the derived
allele. An ``N`` indicates a missing genotype at that position. Finally,
the ``+`` in column seven indicates that individual three possessed the
dominant allele on one chromosome, and had a missing observation on the
other chromosome (this would be coded as ``0/.`` in a VCF).

The SMC++ format for this input file is::

    1   0   2   4
    1   0   0   2
    1   1   0   4
    2   0   0   4
    1   -1  0   2
    1   0   0   3
    2   0   0   0
    1   2   1   4


Output Data Format
==================
Upon completion, SMC++ will write a `JSON-formatted
<https://en.wikipedia.org/wiki/JSON>`_ model file into the into the
analysis directory. The file is human-readable and contains various
parameters related to the fitting procedure.

Upon completion, SMC++ will output a tab-delimited table containing
the estimation results. The three columns `a`, `b`, and `s` define a
piecewise population model such that the estimated effective population
size `s` generations in the past, `eta(s)`, is:::

    eta(s) = a[i] * exp(log(b[i]/a[i])/(s[i] - s[i-1]) * (s - s[i-1])), s[i-1] <= s < s[i],

where we define `s[0] = 0` by convention. Note that the population      
sizes `a` and `b` are the *diploid* effective population size at each   
corresponding time interval.                                            
