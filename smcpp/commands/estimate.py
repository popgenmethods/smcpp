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
from .. import _smcpp, util, em_context as ctx
from ..model import SMCModel

np.set_printoptions(linewidth=120, suppress=True)
logger = logging.getLogger(__name__)

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
    parser.add_argument("-o", "--outdir", help="output directory", default=".", widget="DirChooser")
    parser.add_argument('-v', '--verbose', action='store_true', help="generate tremendous amounts of output")
    pop_params.add_argument('--N0', default=1e4, type=float, help="reference effective (diploid) population size to scale output.")
    pop_params.add_argument('mu', type=float, help="per-generation mutation rate")
    pop_params.add_argument('r', type=float, help="per-generation recombination rate")
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format", widget="MultiFileChooser")

def _obsfs_helper(args):
    ol, n = args
    obsfs = np.zeros([3, n - 1])
    olsub = ol[np.logical_and(ol[:, 1:3].min(axis=1) != -1, ol[:, -1] == n - 2)]
    for a in [0, 1, 2]:
        for b in range(n - 1):
            obsfs[a, b] = olsub[np.logical_and(olsub[:, 1] == a, olsub[:, 2] == b)][:, 0].sum()
    return obsfs

def _thin_helper(args):
    thinned = np.array(_smcpp.thin_data(*args), dtype=np.int32)
    return util.compress_repeated_obs(thinned)

def regularizer(y, coords, cons):
    ## Regularizer
    reg = 0
    dreg = np.zeros(len(coords))
    aa, bb = y
    cs = np.cumsum(ctx.model.s)
    for i in range(1, ctx.model.K):
        x = bb[i - 1] - aa[i]
        _cons = cons
        # rr = (abs(x) - .25) if abs(x) >= 0.5 else x**2
        reg += _cons * x**2
        for c in [(0 if i - 1 in ctx.flat_pieces else 1, i - 1), (0, i)]:
            dx = 1 if c[1] == i - 1 else -1
            try:
                i = coords.index(c)
                dreg[i] += _cons * 2 * x * dx
            except ValueError:
                pass
    return reg, dreg

def pretrain(args, obsfs):
    n = obsfs.shape[1] + 1
    coords = [(u, v) for v in range(ctx.model.K) for u in ([0] if v in ctx.flat_pieces else [0, 1])]
    a = np.ones(ctx.model.K)
    b = np.ones(ctx.model.K)
    uobsfs = util.undistinguished_sfs(obsfs)
    logger.debug(uobsfs)
    def f(x):
        y = np.array([a, b])
        for cc, xx in zip(coords, x):
            y[cc] = xx
        y[1, ctx.flat_pieces] = y[0, ctx.flat_pieces]
        print(y)
        sfs, jac = _smcpp.sfs(n, (y[0], y[1], ctx.model.s), 0., 49.0, ctx.model.theta, coords)
        usfs = util.undistinguished_sfs(sfs)
        ujac = util.undistinguished_sfs(jac)
        kl = -(uobsfs * np.log(usfs)).sum()
        dkl = -(uobsfs[:, None] * ujac / usfs[:, None]).sum(axis=0)
        ret = [kl, dkl]
        reg, dreg = regularizer(y, coords, ctx.lambda_penalty * 1e-3)
        ret[0] += reg
        ret[1] += dreg
        logger.debug(ret[0])
        logger.debug(ret[1])
        logger.debug("regularizer: %s" % str((reg, dreg)))
        return ret
    x0 = np.ones(len(coords))
    res = scipy.optimize.fmin_tnc(f, x0, None,
            bounds=[tuple(ctx.bounds[cc]) for cc in coords])
    ret = np.ones([2, ctx.model.K])
    for cc, xx in zip(coords, res[0]):
        ret[cc] = xx
    ret[1, ctx.flat_pieces] = ret[0, ctx.flat_pieces]
    logger.debug("pretraining results: %s" % str(ret))
    return ret

def extract_pieces(piece_str):
    '''Convert PSMC-style piece string to model representation.'''
    pieces = []
    for piece in piece_str.split("+"):
        try:
            num, span = list(map(int, piece.split("*")))
        except ValueError:
            span = int(piece)
            num = 1
        pieces += [span] * num
    return pieces

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
    smcpp_data = util.parse_text_datasets(args.data)
    n = smcpp_data['n']
    logger.debug("max samples: %d" % n)

    ## Calculate observed SFS for use later
    pool = multiprocessing.Pool(None)
    osfs = list(pool.map(_obsfs_helper, [(ob, n) for ob in smcpp_data['obs']]))
    obsfs = np.sum(osfs, axis=0)
    obsfs /= obsfs.sum()
    logger.debug("Observed SFS:\n%s", str(obsfs))

    if args.thinning is not None:
        smcpp_data['obs'] = pool.map(_thin_helper, [(ob, args.thinning, i) for i, ob in enumerate(smcpp_data['obs'])])

    pool.close()
    pool.terminate()
    pool = None

    ## Break up regions separated by lots of missing data,
    ## which are almost independent anyways
    ctx.obs_list = []
    obs_attributes = {}
    for fn, obs in zip(args.data, smcpp_data['obs']):
        long_spans = np.where(obs[:, 0] >= args.span_cutoff)[0]
        cob = 0
        logging.debug("Long spans: %s" % str(long_spans))
        positions = np.insert(np.cumsum(obs[:, 0]), 0, 0)
        for x in long_spans:
            if not np.all(obs[x, 1:] == [-1, 0, 0]):
                logger.warn("Data set contains a very long span of non-missing observations.")
            s = obs[cob:x, 0].sum()
            if s > args.length_cutoff:
                ctx.obs_list.append(np.insert(obs[cob:x], 0, [1, -1, 0, 0], 0))
                sums = ctx.obs_list[-1].sum(axis=0)
                s2 = ctx.obs_list[-1][:,1][ctx.obs_list[-1][:,1]>=0].sum()
                obs_attributes.setdefault(fn, []).append((positions[cob], positions[x], sums[0], 1. * s2 / sums[0], 1. * sums[2] / sums[0]))
            else:
                logger.warn("omitting sequence length < %d as less than length cutoff" % s)
            cob = x + 1
        s = obs[cob:, 0].sum()
        if s > args.length_cutoff:
            ctx.obs_list.append(np.insert(obs[cob:], 0, [1, -1, 0, 0], 0))
            sums = ctx.obs_list[-1].sum(axis=0)
            s2 = ctx.obs_list[-1][:,1][ctx.obs_list[-1][:,1]>=0].sum()
            obs_attributes.setdefault(fn, []).append((positions[cob], positions[-1], sums[0], 1. * s2 / sums[0], 1. * sums[2] / sums[0]))
        else:
            logger.warn("omitting sequence length < %d as less than length cutoff" % s)

    ## Sanity check
    logging.debug("Average hetorozygosity (derived / total bases) by data set:")
    for fn in sorted(obs_attributes):
        logging.debug(fn + ":")
        for attr in obs_attributes[fn]:
            logging.debug("%15d%15d%15d%12g%12g" % attr)

    ## Extract pieces from piece string
    pieces = extract_pieces(args.pieces)

    # Construct time intervals from pieces
    args.t1 /= 2 * args.N0
    args.tK /= 2 * args.N0
    s = np.concatenate([[0.], np.logspace(np.log10(args.t1), np.log10(args.tK), sum(pieces))])
    s = s[1:] - s[:-1]
    sp = np.zeros(len(pieces))
    count = 0
    for i, p in enumerate(pieces):
        sp[i] = s[count:(count+p)].sum()
        count += p
    s = sp

    ctx.model = SMCModel()
    ctx.model.x = np.ones([3, len(s)])
    ctx.model.s = s
    ctx.model.N0 = args.N0
    ctx.model.theta = 2 * args.N0 * args.mu
    ctx.model.rho = 2 * args.N0 * args.r

    ctx.lambda_penalty = args.lambda_penalty
    logger.debug("time points in coalescent scaling:\n%s", str(s))

    ## Initialize model values
    ctx.flat_pieces = [i for i in range(ctx.model.K) if i not in args.exponential_pieces]
    ctx.model.b = ctx.model.a + 0.1
    ctx.model.b[ctx.flat_pieces] = ctx.model.a[ctx.flat_pieces]

    args.Nmin /= 2 * args.N0
    args.Nmax /= 2 * args.N0
    ctx.bounds = np.array([[args.Nmin, args.Nmax]] * ctx.model.K + 
            [[1.01 * args.Nmin, 0.99 * args.Nmax]] * ctx.model.K).reshape([2, ctx.model.K, 2])

    ## Pre-train based on observed SFS
    if not args.no_pretrain:
        logger.info("Initializing model")
        ctx.model.a, ctx.model.b = pretrain(args, obsfs)

    ## Compute hidden states
    hs = _smcpp.balance_hidden_states(ctx.model.x, args.M)
    if hs[0] != 0:
        raise Exception("First hidden state interval must begin at 0")
    cs = np.cumsum(ctx.model.s)
    cs = cs[cs <= hs[1]]
    hs = np.sort(np.unique(np.concatenate([cs, hs])))
    ctx.model.hidden_states = hs
    logger.debug("hidden states:\n%s", str(ctx.model.hidden_states))

    ## Create inference object which will be used for all further calculations.
    ctx.im = _smcpp.PyInferenceManager(n - 2, ctx.obs_list, 
            ctx.model.hidden_states, ctx.model.theta, ctx.model.rho)
    ctx.im.set_params(ctx.model.x, False)
    ctx.im.E_step()
    ctx.llold = -np.inf

    ## Optimization stuff 
    i = 0
    coords = [(aa, j) for j in range(ctx.model.K) for aa in ((0,) if j in ctx.flat_pieces else (0, 1))]
    # Vector of "preconditioners" helps with optimization
    ctx.precond = {coord: 1. / ctx.model.s[coord[1]] for coord in coords}
    # ctx.precond = {coord: 1. for coord in coords}
    if (ctx.model.K - 1, 0) in coords:
        ctx.precond[(ctx.model.K - 1, 0)] = 1. / (15.0 - np.sum(ctx.model.s))
    while i < args.em_iterations:
        logger.info("EM iteration %d/%d" % (i, args.em_iterations))
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
        logger.info("\tNew/old loglik: %f/%f\t(%g%% improvement)" % (ll, ctx.llold, 100. * (ll - ctx.llold) / ctx.llold))
        if ll < ctx.llold:
            logger.warn("Log-likelihood decreased")
        ctx.llold = ll
        esfs = _smcpp.sfs(n, ctx.model.x, 0.0, ctx.model.hidden_states[-1], ctx.model.theta, False)
        write_model(os.path.join(args.outdir, "model.%d.txt" % i))
        logger.debug("model sfs:\n%s" % str(esfs))
        logger.debug("observed sfs:\n%s" % str(obsfs))
        i += 1
    write_model(os.path.join(args.outdir, "model.final.txt"))
