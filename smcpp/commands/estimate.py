'Fit SMC++ to data using the EM algorithm'

from __future__ import division, print_function
import numpy as np
import scipy.optimize
import pprint
import multiprocessing
import sys
import itertools
import sys
import time
import logging
import os
import traceback

# Package imports
from .. import _smcpp, util, estimation_tools
from ..model import SMCModel

np.set_printoptions(linewidth=120, suppress=True)

def init_parser(parser):
    '''Configure parser and parse args.'''
    # FIXME: argument groups not supported in subparsers
    pop_params = parser # parser.add_argument_group('population parameters')
    model = parser # parser.add_argument_group('model')
    hmm = parser # parser.add_argument_group('HMM and fitting parameters')
    model.add_argument('--pieces', type=str, help="span of model pieces", default="32*1")
    model.add_argument('--t1', type=float, help="end-point of first piece, in generations", default=400.)
    model.add_argument('--tK', type=float, help="end-point of last piece, in generations", default=40000.)
    model.add_argument('--exponential-pieces', type=int, action="append", default=[], help="pieces which have exponential growth")
    hmm.add_argument('--thinning', help="emit full SFS every <k>th site", default=10000, type=int, metavar="k")
    hmm.add_argument('--no-pretrain', help="do not pretrain model", action="store_true", default=False)
    hmm.add_argument('--M', type=int, help="number of hidden states", default=32)
    hmm.add_argument('--em-iterations', type=float, help="number of EM steps to perform", default=20)
    hmm.add_argument('--lambda-penalty', type=float, help="regularization penalty", default=.01)
    hmm.add_argument('--lbfgs-factor', type=float, help="stopping criterion for optimizer", default=1e9)
    hmm.add_argument('--Nmin', type=float, help="Lower bound on effective population size", default=1000)
    hmm.add_argument('--Nmax', type=float, help="Upper bound on effective population size", default=400000)
    hmm.add_argument('--span-cutoff', help="treat spans > as missing", default=50000, type=int)
    hmm.add_argument('--length-cutoff', help="omit sequences < cutoff", default=1000000, type=int)
    parser.add_argument("-p", "--second-population", nargs="+", widget="MultiFileChooser", 
            help="Estimate divergence time using data set(s) from a second subpopulation")
    parser.add_argument("-o", "--outdir", help="output directory", default="/tmp", widget="DirChooser")
    parser.add_argument('-v', '--verbose', action='store_true', help="generate tremendous amounts of output")
    pop_params.add_argument('--N0', default=1e4, type=float, help="reference effective (diploid) population size to scale output.")
    pop_params.add_argument('mu', type=float, help="per-generation mutation rate")
    pop_params.add_argument('r', type=float, help="per-generation recombination rate")
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format", widget="MultiFileChooser")

def optimize(coords, factr):
    def fprime(x):
        x0c = ctx.model.x.copy()
        # Preconditioner (sort of)
        for xx, cc in zip(x, coords):
            x0c[cc] = xx * ctx.precond[cc]
        aa, bb, _ = x0c
        bb[ctx.flat_pieces] = aa[ctx.flat_pieces]
        ctx.im.set_params((aa, bb, ctx.model.s), coords)
        res = ctx.im.Q()
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        reg, dreg = regularizer([aa, bb], coords, ctx.lambda_penalty)
        ret[0] += reg
        ret[1] += dreg
        dary = np.zeros([2, ctx.model.K])
        for i, cc in enumerate(coords):
            ret[1][i] *= ctx.precond[cc]
            dary[cc] = ret[1][i]
        logger.debug(x0c)
        logger.debug(dary)
        logger.debug(ret[0])
        logger.debug("regularizer: %s" % str((reg, dreg)))
        return ret
    # logger.debug("gradient check")
    # xx0 = np.array([ctx.x[cc] / ctx.precond[cc] for cc in coords])
    # f0, fp = fprime(xx0)
    # for i, cc in enumerate(coords):
    #     x0c = xx0.copy()
    #     x0c[i] += 1e-8
    #     f1, _ = fprime(x0c)
    #     logger.debug((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
    res = scipy.optimize.fmin_l_bfgs_b(fprime, [ctx.model.x[cc] / ctx.precond[cc] for cc in coords], 
            None, bounds=[tuple(ctx.bounds[cc] / ctx.precond[cc]) for cc in coords], 
            factr=factr)
    ret = np.array([x * ctx.precond[cc] for x, cc in zip(res[0], coords)])
    return ret

def write_model(fn):
    open(fn, "wt").write(ctx.model.to_json())

def main(args):
    'Main control loop for EM algorithm'
    ## Create output directory and dump all values for use later
    try:
        os.makedirs(args.outdir)
    except OSError:
        pass # directory exists
    ## Initialize the logger
    # fmt = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logging.addLevelName(logging.DEBUG-1, 'DEBUG1')
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    fmtstr = '%(relativeCreated)d %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, 
            filename=os.path.join(args.outdir, "debug.txt"),
            filemode='wt',
            format=fmtstr)
    sh = logging.StreamHandler()
    if args.verbose:
        sh.setLevel(logging.DEBUG)
    else:
        sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmtstr))
    logging.getLogger().addHandler(sh)

    ## Begin main script
    ## Step 1: load data and clean up a bit
    logger.info("Loading data...")
    datasets_files = [args.data]
    if args.p:
        datasets_files.append(args.p) 
        logger.info("Split estimation mode selected")
    
    ## Parse each data set into an array of observations
    datasets = [util.parse_text_datasets(ds) for ds in datasets_files]

    ## Optionally thin each dataset
    if args.thinning is not None:
        datasets = [estimation_tools.thin_dataset(ds, args.thinning) for ds in datasets]
    
    ## break up long spans
    datasets, attrs = zip(*[estimation_tools.break_long_spans(ds, args.span_cutoff, args.length_cutoff) for ds in datasets])

    ## Sanity check
    for i, ds in enumerate(datasets):
        logging.debug("In population %d:" % i)
        logging.debug("Average heterozygosity (derived / total bases) by data set:")
        for fn, key in zip(datasets_files[i], attrs[i]):
            logging.debug(fn + ":")
            for attr in attrs[i][key]:
                logging.debug("%15d%15d%15d%12g%12g" % attr)

    ## Build time intervals
    t1 = args.t1 / (2 * args.N0)
    tK = args.tK / (2 * args.N0)
    pieces = estimation_tools.extract_pieces(args.pieces)
    time_points = estimation_tools.construct_time_points(t1, tK, pieces)
    logger.debug("time points in coalescent scaling:\n%s", str(time_points))

    ## Construct bounds
    Nmin = args.Nmin / (2 * args.N0)
    Nmax = args.Nmax / (2 * args.N0)
    bounds = np.array([[Nmin, Nmax]] * model.K + 
            [[1.01 * Nmin, 0.99 * Nmax]] * model.K).reshape([2, model.K, 2])

    ## Construct populations
    populations = [Population(ds, time_points, args.exponential_pieces, 
        2 * args.theta * args.N0, 2 * args.rho * args.N0, args.M, bounds, args.pretrain)
        for ds in datasets]
    iserv = InferenceService(populations)

    ## Pre-train based on observed SFS
    if not args.no_pretrain:
        logger.info("Initializing model")
        iserv.pretrain()

    model0 = SMCModel(time_points, args.exponential_pieces)
    iserv.set_params(model0, False)
    iserv.E_step()
    llold = -np.inf

    ## Optimization stuff 
    i = 0
    coords = [(aa, j) for j in range(ctx.model.K) for aa in ((0,) if j in ctx.flat_pieces else (0, 1))]
    # Vector of "preconditioners" helps with optimization
    ctx.precond = {coord: 1. / ctx.model.s[coord[1]] for coord in coords}
    # ctx.precond = {coord: 1. for coord in coords}
    if (ctx.model.K - 1, 0) in coords:
        ctx.precond[(ctx.model.K - 1, 0)] = 1. / (15.0 - np.sum(ctx.model.s))
    while i < args.em_iterations:
        logger.info("EM iteration %d/%d" % (i + 1, args.em_iterations))
        logger.info("\tM-step...")
        ret = optimize(coords, args.lbfgs_factor)
        for xx, cc in zip(ret, coords):
            ctx.model.x[cc] = xx
        ctx.model.b[ctx.flat_pieces] = ctx.model.a[ctx.flat_pieces]
        logger.debug("Current model:\n%s", str(ctx.model.x))
        ctx.im.set_params(ctx.model.x, False)
        logger.info("\tE-step...")
        ctx.im.E_step()
        ll = np.sum(ctx.im.loglik())
        if i > 0:
            logger.info("\tNew/old loglik: %f/%f" % (ll, ctx.llold))
        if ll < ctx.llold:
            logger.warn("Log-likelihood decreased")
        ctx.llold = ll
        esfs = _smcpp.sfs(n, ctx.model.x, 0.0, ctx.model.hidden_states[-1], ctx.model.theta, False)
        write_model(os.path.join(args.outdir, "model.%d.txt" % i))
        logger.debug("model sfs:\n%s" % str(esfs))
        logger.debug("observed sfs:\n%s" % str(obsfs))
        i += 1
    write_model(os.path.join(args.outdir, "model.final.txt"))
