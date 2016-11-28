SMC++ is a program for estimating the size history of populations from
whole genome sequence data.

=================
Quick Start Guide
=================

1. Install the software. See Installation_, below.

2. Convert your VCF(s) to the SMC++ input format with vcf2smc_::

     $ smc++ vcf2smc my.data.vcf.gz out/chr1.smc.gz chr1 Pop1:S1,S2

   This command will parse data for the contig ``chr1`` for samples
   ``S1`` and ``S2`` which are members of population ``Pop1``. You
   should run this once for each independent contig in your dataset,
   producing one SMC++ output file per contig.

3. Fit the model using estimate_::

     $ smc++ estimate --theta .00025 -o analysis/ out/example.chr*.smc.gz

   Depending on sample size and your machine, the fitting procedure
   should take between a few minutes and a few hours. The fitted model
   will be stored in JSON format in ``analysis/model.final.json``. For
   details on the format, see below.

4. Visualize the results using plot_::

     $ smc++ plot analysis/model.final.json plot.pdf

============
Installation
============

SMC++ is installed using ``pip``, the Python package manager::

     $ pip install git+https://github.com/terhorst/smcpp.git

Depending on your platform, ``pip`` will either download a pre-compiled
binary, or compile SMC++ from scratch.

Requirements
============

SMC++ requires the following libraries and executables in order to run:

- Python 2.7 or greater.
- gmp_, for some rational field computations.
- mpfr_, for some extended precision calculations.
- gsl_, the GNU Scientific Library.

Experimental pre-built binaries are available for Unix and Mac OS X
systems. They will download automatically using ``pip`` (see above)
if available for your system. Note that you will still need to have
``libgmp``, ``libgsl`` and ``libmpfr`` accessible on your system in order 
to run SMC++.

.. _Homebrew: http://brew.sh
.. _gmp: http://gmplib.org
.. _mpfr: http://mpfr.org
.. _gsl: https//www.gnu.org/software/gsl/


Installing the Dependencies
===========================

On Ubuntu (or Debian) Linux, the library requirements may be installed
using the commmand::

    $ sudo apt-get install -y libgmp-dev libmpfr-dev libgsl0-dev

On OS X, the easiest way to install them is using Homebrew_::

    $ brew install mpfr gmp gsl

Compilation
===========

If binaries are not available for your platform, ``pip`` will attempt
to compile SMC++. In addition to the above Requirements_, SMC++
nedds a compiler which supports C++11 (e.g. GCC 4.8 or later) *and*
OpenMP_.

Note for OS X users
-------------------
Versions of Clang shipping with Mac OS X do not currently support
OpenMP. For this reason it is recommended that you use gcc instead.
In order to tell ``pip`` to use gcc, set the ``CC`` and ``CXX``
environment variables, e.g.::

    $ CC=gcc-5 CXX=g++-5 pip install git+https://github.com/terhorst/smcpp.git

.. _OpenMP: http://openmp.org

Virtual Environment
===================

SMC++ pulls in a fair number of Python dependencies. If you prefer to
keep this separate from your main Python installation, or do not have
root access on your system, you may wish to install SMC++ inside of a
`virtual environment`_. To do so, first create and activate the virtual
environment::

    $ virtualenv -p python2.7 <desired location>
    $ source <desired location>/bin/active

Then, install SMC++ using ``pip`` as described above.

.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/

=====
Usage
=====

SMC++ comprises several subcommands which are accessed using the syntax::
    
    $ smc++ <subcommand>

vcf2smc
=======
This subcommand converts (biallelic, diploid) VCF data to the format used by
SMC++. 

Required arguments
------------------

1. An `indexed VCF file <http://www.htslib.org/doc/tabix.html>`_.
2. An output file. Appending the ``.gz`` extension will cause the output
   to be compressed; the estimate_ command can read from both compressed
   and uncompressed data sources.
3. A contig name. Each call to vcf2smc_ processes a single contig. 
   VCFs containing multiple contigs should be processed via multiple
   separate runs.
4. A list of population(s) and samples. Each population has an id followed
   by a comma-separated list of sample IDs (column names in the VCF). Up to
   two populations are supported.

For example, to convert contig ``chr1`` of ``vcf.gz`` using samples
``NA12878`` and ``NA12879`` of population ``CEU``, saving to
``chr1.smc.gz``, use::

    $ smc++ vcf2smc vcf.gz chr1.smc.gz chr1 CEU:NA12878,NA12879

Optional arguments
------------------
- ``-d``.  SMC++ relies crucially on the notion of a pair of *distinguished lineages*
  (see paper for details on this terminology). The identity of the
  distinguished lineages is set using the ``-d`` option, which specifies
  the sample(s) which will form the distinguished pair. ``-d`` accepts to
  sample ids. The first allele will be taken from sample 1 and the second
  from sample 2. To form the distinguished pair using one
  haplotype from each of ``NA1287{8,9}`` using the above example::
  
      $ smc++ vcf2smc -d NA12878 NA12879 vcf.gz chr1.smc.gz chr1 CEU:NA12878,NA12879
  
  Note that "first" and "second" allele have no meaning for unphased data!
  
  By varying ``-d`` over the same VCF, the user can create distinct data
  sets for estimation. This is useful for forming composite likelihoods.
  For example, the following command will create three data sets from
  contig ``chr1`` of ``myvcf.gz``, by varying the identity of the distinguished
  individual and treating the remaining two samples as "undistinguished":
  
  .. code-block:: bash
  
      for i in {7..9}; 
          do smc++ vcf2smc -d NA1287$i NA1287$i myvcf.gz out.$i.txt chr1 NA12877 NA12878 NA12890; 
      done

Manual conversion
-----------------
``vcf2smc`` targets a common use-case but may not be sufficient for all
users. Those wishing to implement their own custom conversion to the SMC
data format should see the `input data format`_ description below.

estimate
========

This command will fit a population size history to data. The basic usage
is::

    $ smc++ estimate -o out data.smc.gz

Recommended arguments
---------------------

- ``--theta`` specifies the population-scaled mutation rate, that is
  :math:`2 N_0 \mu` where :math:`\mu` denotes the per-generation
  mutation rate, and :math:`N_0` is the baseline diploid effective
  population size (see ``--N0``, below). If ``-theta`` is not specified,
  Watterson's estimator will be used. It is recommended to set this
  using prior knowledge of :math:`\mu` if at all possible.



plot
====

This command plots fitted size histories.

split
=====

This command fits two-population split models using marginal estimates
produced by estimate_.

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
