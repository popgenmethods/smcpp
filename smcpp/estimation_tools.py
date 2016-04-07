'Miscellaneous estimation and data-massaging functions.'
from __future__ import absolute_import, division, print_function
import numpy as np
from logging import getLogger
logger = getLogger(__name__)
import scipy.optimize
import multiprocessing
import ad.admath

from . import _smcpp, util

## 
## Construct time intervals stuff
## 
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

def construct_time_points(t1, tK, pieces):
    logger.debug((t1, tK, pieces))
    s = np.logspace(np.log10(t1[-1]), np.log10(tK), sum(pieces) + 1)
    s = s[1:] - s[:-1]
    time_points = np.zeros(len(pieces))
    count = 0
    for i, p in enumerate(pieces):
        time_points[i] = s[count:(count+p)].sum()
        count += p
    return np.concatenate([t1, time_points])

##
## Regularization
##
def regularizer(model, penalty, f):
    ## Regularizer
    reg = 0
    for i in range(1, model.K):
        x = model[1, i - 1] - model[0, i]
        reg += regularizer._regs[f](x)
        if model[0, i - 1] != model[1, i - 1]:
            a, b = model[:2, i - 1]
            reg += abs(a - b) 
    return penalty * reg

def _diffabs(x):
    K = 1.
    return 2. / K * ad.admath.log1p(ad.admath.exp(K * x)) - x - 2. / K * ad.admath.log(2)
regularizer._regs = {
        'abs': _diffabs,
        'quadratic': lambda x: x**2
        }

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
    
def pretrain(model, sample_csfs, bounds, theta0, penalizer):
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
        sfs = _smcpp.raw_sfs(model, n, 0., _smcpp.T_MAX, True)
        sfs[0, 0] = 0
        sfs *= theta0
        sfs[0, 0] = 1. - sfs.sum()
        logger.debug("done")
        usfs = util.undistinguished_sfs(sfs)
        kl = -(sample_sfs * ad.admath.log(usfs)).sum()
        reg = penalizer(model)
        kl += reg
        ret = (kl.x, np.array(list(map(kl.d, x))))
        logger.debug("\n%s" % np.array_str(np.array([[float(y) for y in row] for row in model._x]), precision=3))
        logger.debug((reg, ret))
        return ret
    x0 = [float(model[cc]) for cc in model.coords]
    res = scipy.optimize.fmin_tnc(f, 
            x0,
            None,
            bounds=[tuple(bounds[cc]) for cc in coords],
            xtol=.01, disp=False)
    for cc, xx in zip(coords, res[0]):
        model[cc] = xx 
    logger.info("pre-trained model:\n%s" % np.array_str(model.x, precision=2))
    return _smcpp.raw_sfs(model, n, 0., _smcpp.T_MAX, False)

def break_long_spans(dataset, span_cutoff, length_cutoff):
    obs_list = []
    obs_attributes = {}
    for fn, obs in enumerate(dataset):
        long_spans = np.where(obs[:, 0] >= span_cutoff)[0]
        cob = 0
        logger.debug("Long spans: \n%s" % str(obs[long_spans]))
        positions = np.insert(np.cumsum(obs[:, 0]), 0, 0)
        for x in long_spans:
            if not np.all(obs[x, 1:] == [-1, 0, 0]):
                logger.warn("Data set contains a very long span of non-missing observations.")
            s = obs[cob:x, 0].sum()
            if s > length_cutoff:
                obs_list.append(np.insert(obs[cob:x], 0, [1, -1, 0, 0], 0))
                sums = obs_list[-1].sum(axis=0)
                s2 = obs_list[-1][:,1][obs_list[-1][:,1]>=0].sum()
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
