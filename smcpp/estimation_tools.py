'Miscellaneous estimation and data-massaging functions.'

import numpy as np
import logging
logger = logging.getLogger(__name__)
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

def regularizer(model, penalty):
    ## Regularizer
    reg = 0
    cs = np.cumsum(model.s)
    for i in range(1, model.K):
        x = model.b[i - 1] - model.a[i]
        cons = penalty
        # rr = (abs(x) - .25) if abs(x) >= 0.5 else x**2
        reg += cons * x**2
    return reg

def empirical_sfs(obs, n):
    ret = np.zeros([3, n - 1])
    sub = obs[np.logical_and(obs[:, 1:3].min(axis=1) != -1, obs[:, -1] == n - 2)]
    for a in [0, 1, 2]:
        for b in range(n - 1):
            ret[a, b] = sub[np.logical_and(sub[:, 1] == a, sub[:, 2] == b)][:, 0].sum()
    return ret

def _thin_helper(args):
    thinned = np.array(_smcpp.thin_data(*args), dtype=np.int32)
    return util.compress_repeated_obs(thinned)

def thin_dataset(dataset, thinning):
    p = multiprocessing.Pool()
    ret = p.map(_thin_helper, [(chrom, thinning, i) for i, chrom in enumerate(dataset)])
    p.close()
    p.join()
    p.terminate()
    return ret
    
def pretrain(model, obsfs, bounds, theta, penalty):
    '''Pre-train model by fitting to observed SFS. Changes model in place!'''
    logging.debug("pretraining")
    n = obsfs.shape[1] + 1
    fp = model.flat_pieces
    K = model.K
    coords = [(u, v) for v in range(K) for u in ([0] if v in fp else [0, 1])]
    uobsfs = util.undistinguished_sfs(obsfs)
    def f(x):
        for cc, xx in zip(coords, x):
            model[cc] = xx
        sfs = _smcpp.sfs(n, model, 0., _smcpp.T_MAX, theta, True)
        usfs = util.undistinguished_sfs(sfs)
        kl = -(uobsfs * ad.admath.log(usfs)).sum()
        kl += model.regularizer(penalty)
        ret = (kl.x, np.array([kl.d(model[cc]) for cc in coords]))
        return ret
    res = scipy.optimize.fmin_l_bfgs_b(f, 
            [float(model[cc]) for cc in model.coords], None,
            bounds=[tuple(bounds[cc]) for cc in coords], disp=True, factr=1e4)
    for cc, xx in zip(coords, res[0]):
        model[cc] = xx 
    logging.info("pretrained-model:\n%s" % str(model.x))

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
