'Miscellaneous estimation and data-massaging functions.'
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from logging import getLogger
logger = getLogger(__name__)
import scipy.optimize
import scipy.interpolate
import multiprocessing as mp
import ad.admath, ad.linalg

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
    s = np.logspace(np.log10(t1[-1]), np.log10(tK), sum(pieces) + 1)
    s = s[1:] - s[:-1]
    time_points = np.zeros(len(pieces))
    count = 0
    for i, p in enumerate(pieces):
        time_points[i] = s[count:(count+p)].sum()
        count += p
    return np.concatenate([t1, time_points])

def _thin_helper(args):
    thinned = np.array(_smcpp.thin_data(*args), dtype=np.int32)
    return util.compress_repeated_obs(thinned)

def thin_dataset(dataset, thinning):
    '''Only emit full SFS every <thinning> sites'''
    p = mp.get_context("spawn").Pool()
    ret = list(p.map(_thin_helper, [(chrom, thinning, i) for i, chrom in enumerate(dataset)]))
    p.close()
    p.join()
    p.terminate()
    return ret
    
def pretrain(model, sample_csfs, bounds, theta0, folded, penalty):
    '''Pre-train model by fitting to observed SFS. Changes model in place!'''
    logger.debug("pretraining")
    n = sample_csfs.shape[1] + 1
    def undist(sfs):
        return util.undistinguished_sfs(sfs, folded)
    sample_sfs = undist(sample_csfs)
    K = model.K
    coords = np.arange(K)
    def f(x):
        x = ad.adnumber(x)
        model[coords] = x
        logger.debug("requesting sfs")
        sfs = _smcpp.raw_sfs(model, n, 0., np.inf)
        sfs[0, 0] = 0
        sfs *= theta0
        sfs[0, 0] = 1. - sfs.sum()
        logger.debug("done")
        usfs = undist(sfs)
        kl = -(sample_sfs * ad.admath.log(usfs)).sum()
        reg = penalty * model.regularizer()
        kl += reg
        ret = (kl.x, np.array(list(map(kl.d, x))))
        logger.debug("\n%s" % np.array_str(model[:].astype('float'), precision=3))
        logger.debug((reg, ret))
        return ret
    x0 = model[coords].astype('float')
    res = scipy.optimize.fmin_tnc(f, 
            x0,
            None,
            bounds=np.log(bounds[0][coords]),
            xtol=0.01,
            disp=False)
    for cc, xx in zip(coords, res[0]):
        model[cc] = xx 
    logger.info("pre-trained model:\n%s" % np.array_str(model[:].astype('float'), precision=2))
    return _smcpp.raw_sfs(model, n, 0., np.inf).astype('float')

def break_long_spans(dataset, rho, length_cutoff):
    # Spans longer than this are broken up
    # FIXME: should depend on rho
    span_cutoff = 100000
    obs_list = []
    obs_attributes = {}
    for fn, obs in enumerate(dataset):
        miss = obs[0].copy()
        miss[:] = 0
        miss[:2] = [1, -1]
        long_spans = np.where(
            (obs[:, 0] >= span_cutoff) &
            (obs[:, 1] == -1) &
            np.all(obs[:, 3::2] == 0, axis=1))[0]
        cob = 0
        logger.debug("Long missing spans: \n%s" % str(obs[long_spans]))
        positions = np.insert(np.cumsum(obs[:, 0]), 0, 0)
        for x in long_spans:
            s = obs[cob:x, 0].sum()
            if s > length_cutoff:
                obs_list.append(np.insert(obs[cob:x], 0, miss, 0))
                sums = obs_list[-1].sum(axis=0)
                s2 = obs_list[-1][:,1][obs_list[-1][:,1]>=0].sum()
                obs_attributes.setdefault(fn, []).append(
                    (positions[cob], positions[x],
                     sums[0], 1. * s2 / sums[0], 1. * sums[2] / sums[0]))
            else:
                logger.info("omitting sequence length < %d as less than length cutoff %d" % (s, length_cutoff))
            cob = x + 1
        s = obs[cob:, 0].sum()
        miss = np.zeros_like(obs[0])
        miss[:2] = [1, -1]
        if s > length_cutoff:
            obs_list.append(np.insert(obs[cob:], 0, miss, 0))
            sums = obs_list[-1].sum(axis=0)
            s2 = obs_list[-1][:,1][obs_list[-1][:,1]>=0].sum()
            obs_attributes.setdefault(fn, []).append((positions[cob], positions[-1], sums[0], 1. * s2 / sums[0], 1. * sums[2] / sums[0]))
        else:
            logger.info("omitting sequence length < %d as less than length cutoff %d" % (s, length_cutoff))
    return obs_list, obs_attributes
