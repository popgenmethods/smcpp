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

# Package imports
import psmcpp._pypsmcpp, psmcpp.lib.util

np.set_printoptions(linewidth=120, precision=6, suppress=True)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

## Helper functions used for multiprocessing
def _norm_help(args):
    logger.info("Normalizing dataset...")
    return psmcpp.lib.util.normalize_dataset(*args)

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
    parser.add_argument("--output-directory", "-o", help="output directory", default=".")
    parser.add_argument('--config', is_config_file=True, help="config file path")
    pop_params.add_argument('--N0', type=float, help="reference effective (diploid) population size", required=True)
    pop_params.add_argument('--mu', type=float, help="per-generation mutation rate", required=True)
    pop_params.add_argument('--rho', type=float, help="per-generation recombination rate", required=True)
    model.add_argument('--pieces', type=str, help="span of model pieces", required=True, default="32*1")
    model.add_argument('--t1', type=float, help="end-point of first piece, in generations", required=True, default=40.)
    model.add_argument('--tK', type=float, help="end-point of last piece, in generations", required=True, default=40000.)
    model.add_argument('--exponential-pieces', type=int, action="append", default=[], help="pieces which have exponential growth")
    hmm.add_argument('--thinning', type=int, help="frequency to emit full SFS (default: 25 * sample size)")
    hmm.add_argument('--M', type=int, help="number of hidden states", required=True, default=32)
    hmm.add_argument('--hM', type=float, help="hidden state cutoff, in generations", required=True, default=8000.)
    hmm.add_argument('--em-iterations', type=float, help="number of EM steps to perform", default=20)
    hmm.add_argument('--lambda-penalty', type=float, help="regularization penalty", default=.01)
    hmm.add_argument('--lbfgs-factor', type=float, help="stopping criterion for optimizer", default=1e10)
    hmm.add_argument('--Nmin', type=float, help="Lower bound on effective population size", default=500)
    hmm.add_argument('--Nmax', type=float, help="Upper bound on effective population size", default=100000)
    parser.add_argument('data', nargs="+", help="data file(s) in SMC++ format")
    return parser, parser.parse_args()

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

def optimize(ctx, coords, factr=1e9):
    logger.debug("Optimizing factr {factr}".format(factr=factr))
    precond = ctx['precond']
    bounds = ctx['bounds']
    flat_pieces = ctx['flat_pieces']
    im = ctx['im']
    def fprime(x):
        x0c = ctx['x'].copy()
        # Preconditioner (sort of)
        for xx, cc in zip(x, coords):
            x0c[cc] = xx * precond[cc[1]]
        aa, bb = x0c
        bb[flat_pieces] = aa[flat_pieces]
        im.setParams((aa, bb, ctx['s']), coords)
        res = im.Q(ctx['lambda_penalty'])
        lls = np.array([ll for ll, jac in res])
        jacs = np.array([jac for ll, jac in res])
        ret = [-np.mean(lls, axis=0), -np.mean(jacs, axis=0)]
        dary = np.zeros([2, ctx['K']])
        for i, cc in enumerate(coords):
            ret[1][i] *= precond[cc[1]]
            dary[cc] = ret[1][i]
        logger.debug(x0c)
        logger.debug(dary)
        logger.debug(ret[0])
        return ret
    # print("gradient check")
    # xx0 = np.array([x0[cc] / precond[cc[1]] for cc in coords])
    # f0, fp = fprime(xx0)
    # for i, cc in enumerate(coords):
    #     x0c = xx0.copy()
    #     x0c[i] += 1e-8
    #     f1, _ = fprime(x0c)
    #     print(i, cc, f1, f0, (f1 - f0) / 1e-8, fp[i])
    res = scipy.optimize.fmin_l_bfgs_b(fprime, [ctx['x'][cc] / precond[cc[1]] for cc in coords], 
            None, bounds=[tuple(bounds[cc] / precond[cc[1]]) for cc in coords], disp=False, factr=factr)
    logger.debug(res)
    return np.array([x * precond[cc[1]] for x, cc in zip(res[0], coords)])

def main():
    '''Main control loop.'''
    parser, args = setup_parser();

    ## Create output directory and dump all values for use later
    try:
        os.makedirs(args.output_directory)
    except OSError:
        pass # directory exists
    parser.print_values(open(os.path.join(args.output_directory, "config.txt"), "wt"))

    ## Begin main script
    ## Step 1: load data and clean up a bit
    try:
        smcpp_data = pickle.load(open(args.data[0], "rb"))
    except:
        smcpp_data = psmcpp.lib.util.parse_text_datasets(args.data)
    n = smcpp_data['n']

    if args.thinning is None:
        args.thinning = 50 * n

    ## In parallel, process data sets
    pool = multiprocessing.Pool(None)
    obs_list = list(pool.map(_norm_help, [(ob, args.thinning) for ob in smcpp_data['obs']]))
    ## Calculate observed SFS for use later
    osfs = list(pool.map(_obsfs_helper, [(ob, n) for ob in obs_list]))
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
    ctx = {'s': s, 'K': len(s), 'lambda_penalty': args.lambda_penalty}
    logger.debug("time points in coalescent scaling:\n%s", str(s))

    ## Initialize model values
    ctx['x'] = np.ones([2, ctx['K']])
    ctx['a'] = ctx['x'][0]
    ctx['b'] = ctx['x'][1]
    flat_pieces = ctx['flat_pieces'] = [i for i in range(ctx['K']) if i not in args.exponential_pieces]
    ctx['b'][:] = ctx['a'] + 0.1
    ctx['b'][flat_pieces] = ctx['a'][flat_pieces]

    ## Compute hidden states
    args.hM /= 2 * args.N0
    hs = exp_quantiles(args.M, args.hM)
    if hs[0] != 0:
        raise Exception("First hidden state interval must begin at 0")
    ctx['hidden states'] = np.unique(np.sort(np.concatenate([hs, np.cumsum(s)])))
    logger.debug("hidden states:\n%s", str(ctx['hidden states']))

    ## Create inference object which will be used for all further calculations.
    im = ctx['im'] = psmcpp._pypsmcpp.PyInferenceManager(n - 2, obs_list, 
            ctx['hidden states'], 4.0 * args.N0 * args.mu, 4.0 * args.N0 * args.rho)
    im.setParams([ctx[x] for x in "abs"], False)
    im.Estep()
    ctx['llold'] = llold = -np.inf

    ## Bounds
    args.Nmin /= 2 * args.N0
    args.Nmax /= 2 * args.N0
    ctx['bounds'] = np.array([[args.Nmin, args.Nmax]] * ctx['K'] + 
            [[1.01 * args.Nmin, 0.99 * args.Nmax]] * ctx['K']).reshape([2, ctx['K'], 2])

    ## Optimization stuff 
    # Vector of "preconditioners" helps with optimization
    ctx['precond'] = 1. / ctx['s']
    ctx['precond'][-1] = 1. / (15.0 - np.sum(ctx['s']))

    i = 0
    coords = [(aa, j) for j in range(ctx['K']) for aa in ((0,) if j in flat_pieces else (0, 1))]
    while i < args.em_iterations:
        ret = optimize(ctx, coords, args.lbfgs_factor)
        for xx, cc in zip(ret, coords):
            ctx['x'][cc] = xx
        ctx['b'][flat_pieces] = ctx['a'][flat_pieces]
        logger.info("************** EM ITERATION %d ***************" % i)
        logger.info("Current model:\n%s", str(ctx['x']))
        # if i == 5:
        #     print("rebalancing hidden states")
        #     args.hM = mp['hidden states'][-1]
        #     hs = im.balance_hidden_states((mp['a'], mp['b'], mp['s']), args.M)
        #     hs[-1] = args.hM
        #     hs = np.unique(np.sort(np.concatenate([hs, np.cumsum(mp['s'])])))
        #     im.hidden_states = hs
        im.setParams((ctx['a'], ctx['b'], ctx['s']), False)
        im.Estep()
        ll = np.sum(im.loglik(0.0))
        logger.info("New/old loglik: %f/%f" % (ll, llold))
        if ll < llold:
            logger.warn("Log-likelihood decreased")
        esfs = psmcpp._pypsmcpp.sfs(n, (ctx['a'], ctx['b'], ctx['s']), 0.0, 
                ctx['hidden states'][-1], 4 * args.N0 * args.mu, False)
        logger.debug("model sfs:\n%s" % str(esfs))
        logger.debug("observed sfs:\n%s" % str(obsfs))
        i += 1

    ## Finally, save output and exit
    with open(os.path.join(args.output_directory, "output.txt"), "wt") as out:
        out.write("# SMC++ output\n")
        out.write("a\tb\ts\n")
        ctx['s'][-1] = np.inf
        np.savetxt(out, np.array([ctx['a'] * 2 * args.N0, ctx['b'] * 2 * args.N0, np.cumsum(ctx['s']) * 2 * args.N0]).T, fmt="%f", delimiter="\t")
        out.seek(0)
        logger.debug(out.read())

if __name__=="__main__":
    main()
