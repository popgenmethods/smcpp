import matplotlib
import matplotlib.style
matplotlib.style.use('seaborn-ticks')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import numpy as np
import scipy.optimize
import scipy.ndimage
import pprint
import multiprocessing
import sys
from collections import Counter
import sys
import json
import os

from .. import _smcpp, util, model, estimation_tools
from . import command
from smcpp.logging import getLogger
logger = getLogger(__name__)

class Posterior(command.Command, command.ConsoleCommand):
    "Store/visualize posterior decoding of TMRCA"

    def __init__(self, parser):
        command.Command.__init__(self, parser)
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
        parser.add_argument("output", metavar="arrays.npz", 
                            help="location to save posterior decoding arrays")
        parser.add_argument("data", type=str, nargs="+", 
                metavar="data.smc[.gz]", help="SMC++ data set(s) to decode")
        hmm = parser.add_argument_group("HMM parameters")
        hmm.add_argument(
            '--M', type=int, help="number of hidden states",
            default=32)

    def main(self, args):
        command.Command.main(self, args)
        if args.colorbar and not args.heatmap:
            logger.error("Can't specify --colorbar without --heatmap")
            sys.exit(1)
        j = json.load(open(args.model, "rt"))
        klass = getattr(model, j['model']['class'])
        m = klass.from_dict(j['model'])
        files = estimation_tools.files_from_command_line_args(args.data)
        contigs = estimation_tools.load_data(files)
        if len(set(c.key for c in contigs)) > 1:
            logger.error("All data sets must be from same population and have same sample size")
            sys.exit(1)
        hidden_states = estimation_tools.balance_hidden_states(
            m.distinguished_model, args.M + 1) / (2. * m.distinguished_model.N0)
        logger.debug("hidden states (balanced w/r/t model): %s", np.array(hidden_states).round(3))
        all_obs = []
        n = a = None
        for contig in contigs:
            obs = contig.data
            if ((n is not None and np.any(contig.n != n)) or
                (a is not None and np.any(contig.a != a))):
                logger.error("Mismatch between n/a from different contigs")
                sys.exit(1)
            n = contig.n
            a = contig.a
            npop = obs.shape[1] // 2 - 1
            assert len(n) == npop

            lb = 0 if args.start is None else args.start
            ub = obs[:, 0].sum() if args.end is None else args.end
            ## FIXME? Due to the compressed input format the endpoints are only
            ## approximately picked out.
            pos = np.cumsum(obs[:, 0])
            obs = obs[(pos >= lb) & (pos <= ub)]
            obs = np.insert(obs, 0, [[1] + [-1, 0, 0] * npop], 0)
            all_obs.append(obs)
        # Perform thinning, if requested
        if args.thinning > 1:
            all_obs = estimation_tools.thin_dataset(all_obs, [args.thinning] * len(all_obs))
        if npop == 1:
            im = _smcpp.PyOnePopInferenceManager(
                n[0], all_obs, hidden_states, contig.key, args.polarization_error)
        else:
            assert npop == 2
            im = _smcpp.PyTwoPopInferenceManager(
                *n, *a, all_obs, hidden_states, contig.key, args.polarization_error)
        im.theta = j['theta']
        im.rho = j['rho']
        im.save_gamma = True
        im.model = m
        im.E_step()
        gammas = im.gammas
        for g in gammas:
            Lr = g.sum(axis=0)
            g /= Lr
            L = Lr.sum()
        if os.environ.get("SMCPP_DEBUG"):
            import ipdb
            ipdb.set_trace()
        kwargs = {path: g for path, g in zip(args.data, gammas)}
        kwargs.update({path + "_sites": obs[:, 0] for path, obs in zip(args.data, all_obs)})
        np.savez_compressed(args.output, hidden_states=hidden_states, **kwargs)
        if args.heatmap:
            obs = all_obs[0]
            if len(args.data) > 1:
                logger.error("--heatmap is only supported for one data set")
                sys.exit(1)
            # Plotting code
            gamma = g
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
