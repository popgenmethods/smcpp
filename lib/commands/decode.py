from __future__ import division
import numpy as np
import scipy.optimize
import scipy.ndimage
import pprint
import multiprocessing
import sys
import itertools
from collections import Counter
import sys
import itertools as it

from .. import _pypsmcpp, util

def init_parser(parser):
    parser.add_argument("--M", type=int, default=None, 
            help="number of hidden states. If this argument is "
            "specified, hidden states will be calculated to give "
            "a uniform probability of coalescence under the model "
            "stationary distribution. If this argument is not specified, "
            "hidden states will be obtained from <model>.")
    parser.add_argument("--start" type=int, help="base at which to begin posterior decode")
    parser.add_argument("--end" type=int, help="base at which to end posterior decode")
    parser.add_argument("--thinning", type=int, default=1, help="emit full SFS only every <k>th site", metavar="k")
    parser.add_argument("--width", type=int,
            help="number of columns in outputted posterior decoding matrix. If "
            "width < L, matrix will be downsampled prior to saving. "
            "Useful for plotting and/or avoiding huge writes to disk.")
    parser.add_argument("model", type=str, help="SMC++-formatted model to use in forward-backward algorithm")
    parser.add_argument("data", type=str, help="data to decode")
    parser.add_argument("output", type=str, help="destination to save posterior decoding matrix. "
            ".gz enables compression")

def main(args):
    params = {line[2:].strip().split(":") for line in open(args.model, "rt") if line[0] == "#" and ":" in line}
    N0 = float(params['N0'])
    theta = float(params['theta'])
    rho = float(params['rho'])
    a, b, s = np.loadtxt(args.model).T / (2. * float(params['N0']))
    data = util.parse_text_datasets([args.data])
    if args.M is None:
        hidden_states = np.loadtxt(params['hidden states'])
    else:
        hidden_states = _pypsmcpp.balance_hidden_states((a, b, s), args.M)
    M = len(hidden_states) - 1

    obs = data['obs'][0]
    lb = -1 if args.start is None else args.start
    ub = obs[:,0].sum() + 1 if args.end is None else args.end
    ## FIXME? Due to the compressed input format the endpoints are only
    ## approximately picked out.
    pos = np.cumsum(obs[:,0])
    obs = obs[pos >= lb & pos <= ub]
    L = obs.sum(axis=0)

    # Perform thinning, if requested
    if args.thinning is not None:
        obs = util.compress_repeated_obs(_pypsmcpp.thin_data(obs, args.thinning, 0))

    im = psmcpp._pypsmcpp.PyInferenceManager(data['n'], [obs], hidden_states, ctx.theta, ctx.rho)
    im.save_gamma = True
    im.set_params((a, b, s), False)
    im.E_step(True)
    gamma = np.zeros([M, L])
    sp = 0
    for row, col in it.izip(obs, im.gammas[0].T):
        an = row[0]
        gamma[:, sp:(sp+an)] = col[:, None] / an
        sp += an
    if args.width not None:
        logging.info("Downscaling posterior decoding matrix")
        gamma = scipy.ndimage.zoom(gamma, (1.0, 1. * args.width / args.L))
    np.savetxt(args.output, gamma, fmt="%g")
