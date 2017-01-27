.. image:: https://travis-ci.org/popgenmethods/smcpp.svg?branch=master 
    :target: https://travis-ci.org/popgenmethods/smcpp
    
SMC++ is a program for estimating the size history of populations from
whole genome sequence data.

Quick Start Guide
=================

1. Install the software using the `installation instructions`_.
   
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
   will be stored in JSON format in ``analysis/model.final.json``.

4. Visualize the results using plot_::

     $ smc++ plot plot.pdf analysis/model.final.json

SMC++ can also estimate and plot joint demographies from pairs of
populations; see split_.


Installation instructions
=========================

Installer binaries are available from the `releases page`_. Download
the installer for your platform and then run it using ``bash``.
The script will walk you through the installation process. You may
need to ``source /path/to/smcpp/bin/activate`` before running
``/path/to/smcpp/bin/smc++`` in order to prevent conflicts with your
existing Python installation.

The installers are based on the Anaconda_ scientific Python distribution.
If Anaconda already exists on your machine, a more efficient way to
install SMC++ is by using the ``conda`` command::

    $ conda install -c terhorst -c bioconda -c conda-forge smcpp

This will automatically download all necessary dependencies and create
an ``smc++`` executable in the ``bin/`` folder of your Anaconda
distribution.

If neither of these options works for you, you may build the software
from scratch using the `build instructions`_ provided in the next
section.

.. _releases page: https://github.com/popgenmethods/smcpp/releases
.. _Anaconda: https://www.continuum.io/downloads

Build instructions
==================
SMC++ requires the following libraries and executables in order compile and run:

- Python 3.3 or greater.
- A C++-11 compiler (gcc 4.8 or later, for example).
- gmp_, for some rational field computations.
- mpfr_ (at least version 3.0.0), for some extended precision calculations.
- gsl_, the GNU Scientific Library.

On Ubuntu (or Debian) Linux, the library requirements may be installed
using the commmand::

    $ sudo apt-get install -y libgmp-dev libmpfr-dev libgsl0-dev

On OS X, the easiest way to install them is using Homebrew_::

    $ brew install mpfr gmp gsl

After installing the requirements, SMC++ may be built by running::
    
    $ pip install git+https://github.com/popgenmethods/smcpp

.. _Homebrew: http://brew.sh
.. _gmp: http://gmplib.org
.. _mpfr: http://mpfr.org
.. _gsl: https//www.gnu.org/software/gsl/

Note for OS X users
-------------------
Versions of Clang shipping with Mac OS X do not currently support
OpenMP_. In order to build SMC++ on OS X you must use a compiler that
does, such as ``gcc``::

    $ brew install gcc
    $ CC=gcc-6 CXX=g++-6 python setup.py install

.. _OpenMP: http://openmp.org

Virtual Environment
-------------------
SMC++ pulls in a fair number of Python dependencies. If you prefer to
keep this separate from your main Python installation, or do not have
root access on your system, you may wish to install SMC++ inside of a
`virtual environment`_. To do so, first create and activate the virtual
environment::

    $ virtualenv -p python3 <desired location>
    $ source <desired location>/bin/activate

Then, install SMC++ as described above.

.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/

Usage
=====

SMC++ comprises several subcommands which are accessed using the
syntax::

    $ smc++ <subcommand>

where ``<subcommand>`` is one of vcf2smc_, estimate_, plot_, split_, or
cite_.

vcf2smc
-------

This subcommand converts (biallelic, diploid) VCF data to the format
used by SMC++. 

Required arguments
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^
- ``-d``: SMC++ relies crucially on the notion of a pair of *distinguished lineages*
  (see paper for details on this terminology). The identity of the
  distinguished lineages is set using the ``-d`` option, which specifies
  the sample(s) which will form the distinguished pair. ``-d`` accepts to
  sample ids. The first allele will be taken from sample 1 and the second
  from sample 2. To form the distinguished pair using one
  haplotype from each of ``NA1287{8,9}`` using the above example::
  
      $ smc++ vcf2smc -d NA12878 NA12879 vcf.gz chr1.smc.gz chr1 CEU:NA12878,NA12879
  
  Note that "first" and "second" allele have no meaning for unphased data; if your
  data are not phased, it only makes sense to specify a single individual 
  (e.g. ``-d NA12878 NA12878``).

- ``--mask``, ``-m``: This specifies a BED-formatted mask file whose
  positions will be marked as missing data (across all samples) in
  the outputted SMC++ data set. This can be used to delineate large
  uncalled regions (e.g. centromeres) which are often omitted in VCF
  files; without additional information provided by ``--mask``, there
  is no way to distinguish these missing regions from very long runs
  of homozygosity. For finer-grained control of missing data, setting
  individual positions and samples to the missing genotype, ``./.``,
  also works fine. (The point of ``--mask`` is to save the user the
  trouble of emitting millions of rows of missing observations in the
  VCF).

- ``--missing-cutoff``, ``-c``: This is an alternative to ``--mask`` which will
  automatically treat runs of homozgosity longer than ``-c`` base pairs
  as missing. Typically ``-c`` should be set high so as not
  to filter out legitimate long runs of homozyous bases, which are
  informative about recent demography. This is a fairly crude approach
  to filtering and is only recommended for use in cases where using
  ``--mask`` is not possible.
  
Composite Likelihood
^^^^^^^^^^^^^^^^^^^^
By varying ``-d`` over the same VCF, you can create distinct data
sets for estimation. This is useful for forming composite likelihoods.
For example, the following command will create three data sets from
contig ``chr1`` of ``myvcf.gz``, by varying the identity of the distinguished
individual and treating the remaining two samples as "undistinguished":

.. code-block:: bash

    for i in {7..9}; 
        do smc++ vcf2smc -d NA1287$i NA1287$i myvcf.gz out.$i.txt chr1 NA12877 NA12878 NA12890; 
    done

You can then pass these data sets into estimate_::

   $ smc++ estimate -o output/ out.*.txt

SMC++ treats each file ``out.*.txt`` as an independently evolving
sequence (i.e., a chromosome); the likelihood is simply the product
of SMC++ likelihoods over each of the data sets. In the example above
where the data sets are generated from the same chromosome but different
distinguished individuals (different ``-d``), this independence
assumption is violated, leading to a so-called **composite likelihood**.
The advantage of this approach is that it incorporates genealogical
information from additional distinguished individuals into the analysis,
potentially leading to improved estimates. 

Since (a portion of) the computational and memory requirements of SMC++
scale linearly with the total analyzed sequence length, it is generally
advisable to composite over a relatively small number of individuals. In
practice we generally use 2-10 individuals, depending on genome length,
sample size, etc., and have found that this leads to improved estimation
without causing significant degeneracy in the likelihood.

Manual conversion
^^^^^^^^^^^^^^^^^
``vcf2smc`` targets a common use-case but may not be sufficient for all
users. Those wishing to implement their own custom conversion to the SMC
data format should see the `input data format`_ description below.

estimate
--------

This command will fit a population size history to data. The basic usage
is::

    $ smc++ estimate -o out data.smc.gz

Recommended arguments
^^^^^^^^^^^^^^^^^^^^^

- ``-o`` specifies the directory to store the final estimates as well as
  all intermediate files and debugging output.

- ``--theta`` sets the population-scaled mutation rate, that is
  :math:`2 N_0 \mu` where :math:`\mu` denotes the per-generation
  mutation rate, and :math:`N_0` is the baseline diploid effective
  population size (see ``--N0``, below). If ``-theta`` is not specified,
  Watterson's estimator will be used. It is recommended to set this
  using prior knowledge of :math:`\mu` if at all possible.

- ``--rho`` sets the population-scaled recombination rate, that is
  :math:`2 N_0 r` where :math:`r` denotes the per-generation
  recombination rate. If not specified, this will be estimated from the
  data. The estimates should be fairly accurate if the recombination
  rate is not large compared to the mutation rate.

Optional arguments
^^^^^^^^^^^^^^^^^^

A number of other arguments concerning technical aspects of the fitting
procedure exist. To see them, pass the ``-h`` option to ``estimate``.

plot
----

This command plots fitted size histories. The basic usage is::

    $ smc++ plot plot.png model1.json model2.json [...] modeln.json

where ``model*.json`` are fitted models produced by ``estimate``.

Required arguments
^^^^^^^^^^^^^^^^^^

1. An output file-name. The output format is determined by the extension
   (``.pdf``, ``.png``, ``.jpeg``, etc.)
2. One or more JSON-formatted SMC++ models (the output from estimate_).

Optional arguments
^^^^^^^^^^^^^^^^^^

- ``-g`` sets the generation time (in years) used to scale the x-axis. If not
  given, the plot will be in coalescent units.
- ``--logy`` plots the y-axis on a log scale.
- ``-c`` produces a CSV-formatted table containing the data used to generate
  the plot.

split
-----

This command fits two-population clean split models using marginal
estimates produced by estimate_. To use ``split``, first estimate each
population marginally using ``estimate``::

    $ smc++ vcf2smc my.vcf.gz data/pop1.smc.gz <contig> pop1:ind1_1,ind1_2
    $ smc++ vcf2smc my.vcf.gz data/pop2.smc.gz <contig> pop2:ind2_1,ind2_2
    $ smc++ estimate -o pop1/ <additional options> data/pop1.smc.gz
    $ smc++ estimate -o pop2/ <additional options> data/pop2.smc.gz

Next, create a dataset containing the joint frequency spectrum for both
populations::

    $ smc++ vcf2smc my.vcf.gz data/pop12.smc.gz <contig> pop1:ind1_1,ind1_2 pop2:ind2_1,ind2_2

Finally, run ``split`` to refine the marginal estimates into an estimate
of the joint demography::

    $ smc++ split -o split/ pop1/model.final.json pop2/model.final.json data/*.smc.gz
    $ smc++ plot joint.pdf split/model.final.json

cite
----

This command prints plain- and BibTex-formatted citation information for
the `accompanying paper`_ to the console.

.. _accompanying paper: http://www.nature.com/ng/journal/vaop/ncurrent/ng.3748

File Formats
============

Input Data Format
-----------------
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
------------------
Upon completion, SMC++ will write a `JSON-formatted
<https://en.wikipedia.org/wiki/JSON>`_ model file into the into the
analysis directory. The file is human-readable and contains various
parameters related to the fitting procedure.
