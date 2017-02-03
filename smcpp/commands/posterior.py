import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import seaborn as sns
import numpy as np
import scipy.optimize
import scipy.ndimage
import pprint
import multiprocessing
import sys
from collections import Counter
import sys
import json

from .. import _smcpp, util, model, estimation_tools
from . import command


class Posterior(command.Command):
    "Store/visualize posterior decoding of TMRCA"

    def __init__(self, parser):
        super().__init__(parser)
        command.add_hmm_args(parser)
        parser.add_argument("--start", type=int,
                            help="base at which to begin posterior decode")
        parser.add_argument("--end", type=int,
                            help="base at which to end posterior decode")
        parser.add_argument("--thinning", type=int, default=1,
                help="emit full SFS only every <k>th site. default: 1", metavar="k")
        parser.add_argument("--heatmap",
                            metavar="heatmap.(pdf|png|gif|jpeg)",
                            help="Also draw a heatmap of the posterior TMRCA.")
        parser.add_argument("--colorbar", action="store_true", help="If plotting, add a colorbar")
        parser.add_argument("model", type=str, metavar="model.final.json",
                            help="SMC++ model to use in forward-backward algorithm")
        parser.add_argument("data", type=str, metavar="data.smc[.gz]", 
                            help="SMC++ data set to decode")
        parser.add_argument("output", metavar="arrays.npz", 
                            help="location to save posterior decoding arrays")

    def main(self, args):
        super().main(args)
        if args.colorbar and not args.heatmap:
            logger.error("Can't specify --colorbar without --heatmap")
            sys.exit(1)
        j = json.load(open(args.model, "rt"))
        klass = getattr(model, j['model']['class'])
        m = klass.from_dict(j['model'])
        contig = estimation_tools.load_data([args.data])[0]
        hidden_states = estimation_tools.balance_hidden_states(
            m.distinguished_model, args.M)
        obs = contig.data
        n = np.max(obs[:, 2::2], axis=0)
        npop = obs.shape[1] // 2 - 1
        assert len(n) == npop
        lb = 0 if args.start is None else args.start
        ub = obs[:, 0].sum() if args.end is None else args.end
        ## FIXME? Due to the compressed input format the endpoints are only
        ## approximately picked out.
        pos = np.cumsum(obs[:, 0])
        obs = obs[(pos >= lb) & (pos <= ub)]
        obs = np.insert(obs, 0, [[1, -1] + [0, 0] * npop], 0)
        L = obs.sum(axis=0)[0]

        # Perform thinning, if requested
        if args.thinning > 1:
            obs = estimation_tools.thin_dataset([obs], [args.thinning])[0]
        if npop == 1:
            im = _smcpp.PyOnePopInferenceManager(
                n[0], [obs], hidden_states, contig.key, args.polarization_error)
        else:
            assert npop == 2
            im = _smcpp.PyTwoPopInferenceManager(
                n[0], n[1], [obs], hidden_states, contig.key, args.polarization_error)
        im.theta = j['theta']
        im.save_gamma = True
        im.model = m
        im.E_step()
        gamma = im.gammas[0]
        gamma /= gamma.sum(axis=0)
        # gamma = np.zeros([args.M, L])
        # sp = 0
        # for row, col in zip(obs, im.gammas[0].T):
        #     an = row[0]
        #     gamma[:, sp:(sp + an)] = col[:, None] / an
        #     sp += an
        np.savez_compressed(args.output, hidden_states=hidden_states, 
                            sites=obs[:, 0], posterior=gamma)
        if args.heatmap:
            # Plotting code
            fig, ax = plt.subplots()
            x = np.insert(np.cumsum(obs[:, 0]), 0, 0)
            y = hidden_states[:-1]
            img = NonUniformImage(ax, interpolation="bilinear",
                                  extent=(0, x.max(), y[0], y[-1]))
            img.set_data(x, y, gamma)
            ax.images.append(img)
            ax.set_xlim((0, x.max()))
            ax.set_ylim((y[0], y[-1]))
            if L > 1e7:
                ax.set_xlabel("Position (Mb)")
                fac = 1e-6
            elif L > 1e5:
                ax.set_xlabel("Position (Kb)")
                fac = 1e-3
            else:
                ax.set_xlabel("Position (bp)")
                fac = 1
            label_text = [int(loc * fac) for loc in ax.get_xticks()]
            ax.set_xticklabels(label_text)
            ax.set_ylabel("TMRCA")
            if args.colorbar:
                plt.colorbar(img)
            plt.savefig(args.heatmap)
            plt.close()
