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

from .. import _smcpp, util
from ..model import SMCModel

def init_parser(parser):
    parser.add_argument("--uniform", action="store_true",
            help="Re-compute hidden states such that the probability "
            "of coalescence is uniform with respect to the given model. ")
    parser.add_argument("--start", type=int, help="base at which to begin posterior decode")
    parser.add_argument("--end", type=int, help="base at which to end posterior decode")
    parser.add_argument("--thinning", type=int, default=1, help="emit full SFS only every <k>th site", metavar="k")
    parser.add_argument("--width", type=int,
            help="number of columns in outputted posterior decoding matrix. If "
            "width < L, matrix will be downsampled prior to plotting. ")
    parser.add_argument("--colorbar", action="store_true", help="Add a colorbar")
    parser.add_argument("model", type=str, help="SMC++ model to use in forward-backward algorithm", widget="FileChooser")
    parser.add_argument("data", type=str, help="data to decode", widget="FileChooser")
    parser.add_argument("output", type=str, help="destination to save posterior decoding plot")

def main(args):
    model = SMCModel.from_json(open(args.model, "rt").read())
    data = util.parse_text_datasets([args.data])
    if args.uniform:
        model.hidden_states = _smcpp.balance_hidden_states(model.x, model.M)

    obs = data['obs'][0]
    lb = 0 if args.start is None else args.start
    ub = obs[:,0].sum() if args.end is None else args.end
    ## FIXME? Due to the compressed input format the endpoints are only
    ## approximately picked out.
    pos = np.cumsum(obs[:,0])
    obs = obs[(pos >= lb) & (pos <= ub)]
    obs = np.insert(obs, 0, [[1, -1, 0, 0]], 0)
    L = obs.sum(axis=0)[0]

    # Perform thinning, if requested
    if args.thinning is not None:
        obs = util.compress_repeated_obs(_smcpp.thin_data(obs, args.thinning, 0))

    im = _smcpp.PyInferenceManager(data['n'], [obs], model.hidden_states, model.theta, model.rho)
    im.save_gamma = True
    im.set_params(model.x, False)
    im.E_step()
    gamma = np.zeros([model.M, L])
    sp = 0
    for row, col in it.izip(obs, im.gammas[0].T):
        an = row[0]
        gamma[:, sp:(sp+an)] = col[:, None] / an
        sp += an
    # if args.width is not None:
    #     gamma = scipy.ndimage.zoom(gamma, (1.0, 1. * args.width / L))
    # else:
    #     args.width = L

    # Plotting code
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.image import NonUniformImage
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    x = np.cumsum(obs[:, 0])
    y = model.hidden_states[:-1]
    img = NonUniformImage(ax, interpolation="bilinear", extent=(0, x.max(), y[0], y[-1]))
    g = im.gammas[0]
    g /= g.sum(axis=0)[None, :]
    img.set_data(x, y, g)
    ax.images.append(img)
    ax.set_xlim((0, x.max()))
    ax.set_ylim((y[0], y[-1]))
    # ax.imshow(gamma[::-1], extent=[0, args.width, -0.5, model.M - 0.5], aspect='auto', vmin=0.0)
    label_text = [int(loc * 1e-6) for loc in ax.get_xticks()]
    ax.set_xticklabels(label_text)
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("TMRCA")
    if args.colorbar:
        plt.colorbar(img)
    plt.savefig(args.output)
    plt.close()
