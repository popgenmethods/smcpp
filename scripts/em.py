#/usr/bin/env python2.7
'''Fit SMC++ model to data using the EM algorithm.'''
from __future__ import division
import numpy as np
import scipy.optimize
import pprint
import multiprocessing
import sys
import itertools
import sys
import time
import configargparse
import logging
import os
import traceback

# Package imports
import psmcpp._pypsmcpp, psmcpp.lib.util, psmcpp.lib.plotting, psmcpp.lib.em_context as ctx
from psmcpp.lib.util import undistinguished_sfs

np.set_printoptions(linewidth=120)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

def _obsfs_helper(args):
    ol, n = args
    obsfs = np.zeros([3, n - 1])
    olsub = ol[np.logical_and(ol[:, 1:3].min(axis=1) != -1, ol[:, -1] == n - 2)]
    for r, c1, c2, _ in olsub:
        obsfs[c1, c2] += r
    return obsfs

def exp_quantiles(M, h_M):
    hs = -np.log(1. - np.linspace(0, h_M, M, False) / h_M)
    hs = np.append(hs, h_M)
    hs[0] = 0
    return hs

def setup_parser():
    '''Setup parser and parse args.'''
    parser = configargparse.ArgumentParser("SMC++")
    pop_params = parser.add_argument_group('population parameters')
    model = parser.add_argument_group('model')
    hmm = parser.add_argument_group('HMM and fitting parameters')
    parser.add_argument("-o", "--output-directory", help="output directory", default=".")
    parser.add_argument('-c', '--config', is_config_file=True, help="config file path")
    parser.add_argument('-v', '--verbose', action='store_true', help="output lots of debugging info")
    pop_params.add_argument('--N0', type=float, help="reference effective (diploid) population size", required=True)
    pop_params.add_argument('--mu', type=float, help="per-generation mutation rate", required=True)
    pop_params.add_argument('--rho', type=float, help="per-generation recombination rate", required=True)
    model.add_argument('--pieces', type=str, help="span of model pieces", required=True, default="32*1")
    model.add_argument('--t1', type=float, help="end-point of first piece, in generations", required=True, default=40.)
    model.add_argument('--tK', type=float, help="end-point of last piece, in generations", required=True, default=40000.)
    model.add_argument('--exponential-pieces', type=int, action="append", default=[], help="pieces which have exponential growth")
    hmm.add_argument('--no-pretrain', help="do not pretrain model", action="store_true", default=False)
    hmm.add_argument('--M', type=int, help="number of hidden states", required=True, default=32)
    hmm.add_argument('--em-iterations', type=float, help="number of EM steps to perform", default=20)
    hmm.add_argument('--lambda-penalty', type=float, help="regularization penalty", default=.01)
    hmm.add_argument('--lbfgs-factor', type=float, help="stopping criterion for optimizer", default=1e10)
    hmm.add_argument('--Nmin', type=float, help="Lower bound on effective population size", default=500)
    hmm.add_argument('--Nmax', type=float, help="Upper bound on effective population size", default=100000)
    hmm.add_argument('--length-cutoff', help="omit sequences < cutoff", default=0, type=int)
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format")
    return parser, parser.parse_args()

def pretrain(args, obsfs):
    n = obsfs.shape[1] + 1
    coords = [(u, v) for v in range(ctx.K) for u in ([0] if v in ctx.flat_pieces else [0, 1])]
    a = np.ones(ctx.K)
    b = np.ones(ctx.K)
    uobsfs = undistinguished_sfs(obsfs)
    logger.info(uobsfs)
    def f(x):
        y = np.array([a, b])
        for cc, xx in zip(coords, x):
            y[cc] = xx
        y[1, ctx.flat_pieces] = y[0, ctx.flat_pieces]
        print(y)
        sfs, jac = psmcpp._pypsmcpp.sfs(n, (y[0], y[1], ctx.s), 0., 49.0, 2 * args.N0 * args.mu, coords)
        usfs = undistinguished_sfs(sfs)
        ujac = undistinguished_sfs(jac)
        kl = -(uobsfs * np.log(usfs)).sum()
        dkl = -(uobsfs[:, None] * ujac / usfs[:, None]).sum(axis=0)
        reg = 0
        dreg = np.zeros(dkl.shape)
        for i in range(1, ctx.K):
            x = y[1, i - 1] - y[0, i]
            sx = np.sign(x)
            reg += ctx.lambda_penalty * abs(x)
            for c in [(1, i - 1), (0, i)]:
                try:
                    i = coords.index((0, i))
                    dreg[i] += ctx.lambda_penalty * sx 
                except ValueError:
                    pass
                sx *= sx
        kl += reg
        dkl += dreg
        logger.info((reg, dreg))
        logger.info((kl, usfs))
        return kl, dkl
    x0 = np.ones(len(coords))
    res = scipy.optimize.fmin_tnc(f, x0, None,
            bounds=[tuple(ctx.bounds[cc]) for cc in coords],
            disp=5, xtol=.0001)
    ret = np.ones([2, ctx.K])
    for cc, xx in zip(coords, res[0]):
        ret[cc] = xx
    ret[1, ctx.flat_pieces] = ret[0, ctx.flat_pieces]
    logger.info("pretraining results: %s" % str(ret))
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

def optimize(coords, factr=1e9):
    def fprime(x):
        x0c = ctx.x.copy()
        # Preconditioner (sort of)
        for xx, cc in zip(x, coords):
            x0c[cc] = xx * ctx.precond[cc]
        aa, bb = x0c
        bb[ctx.flat_pieces] = aa[ctx.flat_pieces]
        ctx.im.setParams((aa, bb, ctx.s), coords)
        res = ctx.im.Q()
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        reg, jac = ctx.im.regularizer
        reg *= ctx.lambda_penalty
        jac *= ctx.lambda_penalty
        logger.info("regularizer: %s" % str((reg, jac)))
        ret[0] += reg
        ret[1] += jac
        dary = np.zeros([2, ctx.K])
        for i, cc in enumerate(coords):
            ret[1][i] *= ctx.precond[cc]
            dary[cc] = ret[1][i]
        logger.info(x0c)
        logger.info(dary)
        logger.info(ret[0])
        return ret
    # logger.debug("gradient check")
    # xx0 = np.array([ctx.x[cc] / ctx.precond[cc] for cc in coords])
    # f0, fp = fprime(xx0)
    # for i, cc in enumerate(coords):
    #     x0c = xx0.copy()
    #     x0c[i] += 1e-8
    #     f1, _ = fprime(x0c)
    #     logger.debug((i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i]))
    res = scipy.optimize.fmin_l_bfgs_b(fprime, [ctx.x[cc] / ctx.precond[cc] for cc in coords], 
            None, bounds=[tuple(ctx.bounds[cc] / ctx.precond[cc]) for cc in coords], 
            disp=True, factr=factr)
    return np.array([x * ctx.precond[cc] for x, cc in zip(res[0], coords)])

    ## Ignore below here
    def f2(x):
        x0c = ctx.x.copy()
        # Preconditioner (sort of)
        for xx, cc in zip(x, coords):
            x0c[cc] = xx
        aa, bb = x0c
        bb[ctx.flat_pieces] = aa[ctx.flat_pieces]
        ctx.im.setParams((aa, bb, ctx.s), False)
        res = ctx.im.Q()
        reg = ctx.im.regularizer
        reg *= ctx.lambda_penalty
        logger.info("f2\n%s" % str(np.array([aa,bb])))
        return -np.mean(res) + reg
    def fprime2(x):
        x0c = ctx.x.copy()
        # Preconditioner (sort of)
        for xx, cc in zip(x, coords):
            x0c[cc] = xx
        aa, bb = x0c
        bb[ctx.flat_pieces] = aa[ctx.flat_pieces]
        logger.info("f2\n%s" % str(np.array([aa,bb])))
        try:
            ctx.im.setParams((aa, bb, ctx.s), coords)
        except RuntimeError as e:
            if e.message == "SFS is not a probability distribution":
                return (np.inf, None)
            else:
                raise
        res = ctx.im.Q()
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        # reg, jac = ctx.im.regularizer
        # reg *= ctx.lambda_penalty
        # jac *= ctx.lambda_penalty
        # ret[0] += reg
        # ret[1] += jac
        reg = 0
        dreg = np.zeros(jac.shape)
        for i in range(1, ctx.K - 1):
            x = bb[i - 1] - aa[i]
            sx = np.sign(x)
            reg += ctx.lambda_penalty * abs(x)
            for c in [(1, i - 1), (0, i)]:
                try:
                    i = coords.index((0, i))
                    dreg[i] += ctx.lambda_penalty * sx 
                except ValueError:
                    pass
                sx *= sx
        logger.info("f'2\n%s" % str(ret[1]))
        logger.info("reg\n%s %s" % (str(reg), str(dreg)))
        ret[0] += reg
        ret[1] += dreg
        return ret
    res = scipy.optimize.fmin_tnc(fprime2, 
            [ctx.x[cc] for cc in coords], None,
            bounds=[tuple(ctx.bounds[cc]) for cc in coords],
            disp=5, xtol=.0001, scale=[ctx.precond[cc] for cc in coords])
    logger.debug(res)
    return res[0]

def write_output(tag, args):
## Finally, save output and exit
    fn = os.path.join(args.output_directory, "output.%s.txt" % tag)
    with open(fn, "wt") as out:
        out.write("# SMC++ output\n")
        out.write("# a\tb\ts\n")
        ctx.s[-1] = np.inf
        ret = np.array([ctx.a * 2 * args.N0, ctx.b * 2 * args.N0, np.cumsum(ctx.s) * 2 * args.N0]).T 
        np.savetxt(out, ret, fmt="%f", delimiter="\t")
    return ret

def main():
    '''Main control loop.'''
    parser, args = setup_parser();

    ## Create output directory and dump all values for use later
    try:
        os.makedirs(args.output_directory)
    except OSError:
        pass # directory exists
    parser.print_values(open(os.path.join(args.output_directory, ".config.txt"), "wt"))

    ## Initialize the logger
    fh = logging.FileHandler(os.path.join(args.output_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    fh = logging.FileHandler(os.path.join(args.output_directory, "debug.txt"))
    fh.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(fh)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ## Begin main script
    ## Step 1: load data and clean up a bit
    try:
        smcpp_data = pickle.load(open(args.data[0], "rb"))
    except:
        smcpp_data = psmcpp.lib.util.parse_text_datasets(args.data)
    n = smcpp_data['n']
    logger.debug("max samples: %d" % n)

    ## In parallel, process data sets
    pool = multiprocessing.Pool(None)
    obs_list = ctx.obs_list = [subob 
            for subobs in pool.map(psmcpp.lib.util.break_long_missing_spans, smcpp_data['obs'])
            for subob in subobs 
            if subob[:, 0].sum() > args.length_cutoff]
    ## Calculate observed SFS for use later
    osfs = list(pool.map(_obsfs_helper, [(ob, n) for ob in smcpp_data['obs']]))
    pool.close()
    pool.terminate()
    pool = None
    obsfs = np.sum(osfs, axis=0)
    obsfs /= obsfs.sum()
    logger.info("Observed SFS:\n%s", str(obsfs))

    ## Extract pieces from piece string
    pieces = extract_pieces(args.pieces)

    # Construct time intervals from pieces
    args.t1 /= 2 * args.N0
    args.tK /= 2 * args.N0
    s = np.logspace(np.log10(args.t1), np.log10(args.tK), sum(pieces))
    s = np.concatenate(([args.t1], s[1:] - s[:-1]))
    sp = np.zeros(len(pieces))
    count = 0
    for i, p in enumerate(pieces):
        sp[i] = s[count:(count+p)].sum()
        count += p
    s = sp
    ctx.s = s
    ctx.K = len(s)
    ctx.lambda_penalty = args.lambda_penalty
    logger.info("time points in coalescent scaling:\n%s", str(s))

    ## Initialize model values
    ctx.x = np.ones([2, ctx.K])
    ctx.a = ctx.x[0]
    ctx.b = ctx.x[1]
    ctx.flat_pieces = [i for i in range(ctx.K) if i not in args.exponential_pieces]
    ctx.b[:] = ctx.a + 0.1
    ctx.b[ctx.flat_pieces] = ctx.a[ctx.flat_pieces]

    args.Nmin /= 2 * args.N0
    args.Nmax /= 2 * args.N0
    ctx.bounds = np.array([[args.Nmin, args.Nmax]] * ctx.K + 
            [[1.01 * args.Nmin, 0.99 * args.Nmax]] * ctx.K).reshape([2, ctx.K, 2])

    ## Pre-train based on observed SFS
    if not args.no_pretrain:
        logger.info("Pre-training model")
        ctx.a[:], ctx.b[:] = pretrain(args, obsfs)

    ## Compute hidden states
    hs = psmcpp._pypsmcpp.balance_hidden_states((ctx.a, ctx.b, ctx.s), args.M)
    if hs[0] != 0:
        raise Exception("First hidden state interval must begin at 0")
    ctx.hidden_states = hs
    logger.info("hidden states:\n%s", str(ctx.hidden_states))

    ## Create inference object which will be used for all further calculations.
    ctx.im = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, 
            ctx.hidden_states, 2.0 * args.N0 * args.mu, 2.0 * args.N0 * args.rho)
    ctx.im.setParams([ctx.a, ctx.b, ctx.s], False)
    ctx.im.Estep()
    ctx.llold = -np.inf

    ## Optimization stuff 
    i = 0
    coords = [(aa, j) for j in range(ctx.K) for aa in ((0,) if j in ctx.flat_pieces else (0, 1))]
    # Vector of "preconditioners" helps with optimization
    ctx.precond = {coord: 1. / ctx.s[coord[1]] for coord in coords}
    if (ctx.K - 1, 0) in coords:
        ctx.precond[(ctx.K - 1, 0)] = 1. / (15.0 - np.sum(ctx.s))
    while i < args.em_iterations:
        ret = optimize(coords, args.lbfgs_factor)
        for xx, cc in zip(ret, coords):
            ctx.x[cc] = xx
        ctx.b[ctx.flat_pieces] = ctx.a[ctx.flat_pieces]
        logger.info("************** EM ITERATION %d ***************" % i)
        logger.info("Current model:\n%s", str(ctx.x))
        # if i == 5:
        #     print("rebalancing hidden states")
        #     args.hM = mp['hidden states'][-1]
        #     hs = im.balance_hidden_states((mp['a'], mp['b'], mp['s']), args.M)
        #     hs[-1] = args.hM
        #     hs = np.unique(np.sort(np.concatenate([hs, np.cumsum(mp['s'])])))
        #     im.hidden_states = hs
        ctx.im.setParams((ctx.a, ctx.b, ctx.s), False)
        ctx.im.Estep()
        ll = np.sum(ctx.im.loglik())
        logger.info("New/old loglik: %f/%f" % (ll, ctx.llold))
        if ll < ctx.llold:
            logger.warn("Log-likelihood decreased")
        ctx.llold = ll
        esfs = psmcpp._pypsmcpp.sfs(n, (ctx.a, ctx.b, ctx.s), 0.0, 
                ctx.hidden_states[-1], 2. * args.N0 * args.mu, False)
        write_output(str(i), args)
        logger.info("model sfs:\n%s" % str(esfs))
        logger.info("observed sfs:\n%s" % str(obsfs))
        i += 1
    write_output("final", args)

if __name__=="__main__":
    main()
