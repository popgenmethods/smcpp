'Miscellaneous estimation and data-massaging functions.'

import numpy as np
from logging import getLogger
logger = getLogger(__name__)
import scipy.optimize
import multiprocessing
import ad.admath

from . import _smcpp, util

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

def regularizer(model, penalty, f):
    ## Regularizer
    reg = 0
    cs = np.cumsum(model.s)
    for i in range(1, model.K):
        x = model[1, i - 1] - model[0, i]
        cons = penalty
        # rr = (abs(x) - .25) if abs(x) >= 0.5 else x**2
        reg += cons * regularizer._regs[f](x)
    return reg
def _diffabs(x):
    K = 1.
    return 2. / K * ad.admath.log1p(ad.admath.exp(K * x)) - x - 2. / K * ad.admath.log(2)
regularizer._regs = {
        'abs': _diffabs,
        'quadratic': lambda x: x**2
        }

def empirical_sfs(obs, n):
    ret = np.zeros([3, n - 1])
    sub = obs[np.logical_and(obs[:, 1:3].min(axis=1) != -1, obs[:, -1] == n - 2)]
    for a in [0, 1, 2]:
        for b in range(n - 1):
            ret[a, b] = sub[np.logical_and(sub[:, 1] == a, sub[:, 2] == b)][:, 0].sum()
    return ret

## TODO: move this to util
def _thin_helper(args):
    thinned = np.array(_smcpp.thin_data(*args), dtype=np.int32)
    return util.compress_repeated_obs(thinned)

def thin_dataset(dataset, thinning):
    '''Only emit full SFS every <thinning> sites'''
    p = multiprocessing.Pool()
    ret = p.map(_thin_helper, [(chrom, thinning, i) for i, chrom in enumerate(dataset)])
    p.close()
    p.join()
    p.terminate()
    return ret
    
def pretrain(model, sample_csfs, bounds, theta, penalizer):
    '''Pre-train model by fitting to observed SFS. Changes model in place!'''
    logger.debug("pretraining")
    n = sample_csfs.shape[1] + 1
    sample_sfs = util.undistinguished_sfs(sample_csfs)
    fp = model.flat_pieces
    K = model.K
    coords = [(u, v) for v in range(K) for u in ([0] if v in fp else [0, 1])]
    def f(x):
        x = ad.adnumber(x)
        for cc, xx in zip(coords, x):
            model[cc] = xx
        logger.debug("requesting sfs")
        sfs = _smcpp.sfs(model, n, 0., _smcpp.T_MAX, theta, True)
        logger.debug("done")
        usfs = util.undistinguished_sfs(sfs)
        kl = -(sample_sfs * ad.admath.log(usfs)).sum()
        reg = penalizer(model)
        kl += penalizer(model)
        ret = (kl.x, np.array(list(map(kl.d, x))))
        logger.debug("\n%s" % np.array_str(np.array([[float(y) for y in row] for row in model._x]), precision=3))
        logger.debug(ret)
        return ret
    res = scipy.optimize.fmin_tnc(f, 
            [float(model[cc]) for cc in model.coords], None,
            bounds=[tuple(bounds[cc]) for cc in coords],
            xtol=.01)
    for cc, xx in zip(coords, res[0]):
        model[cc] = xx 
    logger.info("pretrained-model:\n%s" % str(model.x))

def break_long_spans(dataset, span_cutoff, length_cutoff):
    obs_list = []
    obs_attributes = {}
    for fn, obs in enumerate(dataset):
        long_spans = np.where(obs[:, 0] >= span_cutoff)[0]
        cob = 0
        logger.debug("Long spans: %s" % str(long_spans))
        positions = np.insert(np.cumsum(obs[:, 0]), 0, 0)
        for x in long_spans:
            if not np.all(obs[x, 1:] == [-1, 0, 0]):
                logger.warn("Data set contains a very long span of non-missing observations.")
            s = obs[cob:x, 0].sum()
            if s > length_cutoff:
                obs_list.append(np.insert(obs[cob:x], 0, [1, -1, 0, 0], 0))
                sums = ctx.obs_list[-1].sum(axis=0)
                s2 = ctx.obs_list[-1][:,1][ctx.obs_list[-1][:,1]>=0].sum()
                obs_attributes.setdefault(fn, []).append((positions[cob], positions[x], sums[0], 1. * s2 / sums[0], 1. * sums[2] / sums[0]))
            else:
                logger.info("omitting sequence length < %d as less than length cutoff" % s)
            cob = x + 1
        s = obs[cob:, 0].sum()
        if s > length_cutoff:
            obs_list.append(np.insert(obs[cob:], 0, [1, -1, 0, 0], 0))
            sums = obs_list[-1].sum(axis=0)
            s2 = obs_list[-1][:,1][obs_list[-1][:,1]>=0].sum()
            obs_attributes.setdefault(fn, []).append((positions[cob], positions[-1], sums[0], 1. * s2 / sums[0], 1. * sums[2] / sums[0]))
        else:
            logger.info("omitting sequence length < %d as less than length cutoff" % s)
    return obs_list, obs_attributes

def construct_time_points(t1, tK, pieces):
    s = np.concatenate([[0.], np.logspace(np.log10(t1), np.log10(tK), sum(pieces))])
    s = s[1:] - s[:-1]
    time_points = np.zeros(len(pieces))
    count = 0
    for i, p in enumerate(pieces):
        time_points[i] = s[count:(count+p)].sum()
        count += p
    return time_points
