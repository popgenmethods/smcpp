import numpy as np
import json
import sys

from .. import _smcpp, util, logging, data_filter
import smcpp.defaults
from smcpp.optimize.optimizers import SMCPPOptimizer, TwoPopulationOptimizer
from smcpp.optimize.plugins import analysis_saver, parameter_optimizer

logger = logging.getLogger(__name__)

from ..model import SMCModel, SMCTwoPopulationModel

_model_cls_d = {cls.__name__: cls for cls in (SMCModel, SMCTwoPopulationModel)}


class BaseAnalysis:
    "Base class for analysis of population genetic data."

    def __init__(self, files, args):
        # Misc. parameter initialiations
        self._args = args
        if args.cores is not None:
            _smcpp.set_num_threads(args.cores)
        self._N0 = .5e-4 / args.mu  # .0001 = args.mu * 2 * N0
        self._theta = 2. * self._N0 * args.mu
        logger.info("theta: %f", self._theta)
        if args.r is not None:
            self._rho = 2 * self._N0 * args.r
        else:
            self._rho = self._theta
        assert np.all(np.isfinite([self._rho, self._theta]))
        logger.info("rho: %f", self._rho)
        self._penalty = 0.
        self._niter = args.em_iterations
        if args.unfold:
            args.polarization_error = 0.
            logger.warning(
                "Using unfolded SFS. The user should verify "
                "that the ancestral allele has been correctly "
                "coded."
            )
        if args.polarization_error > 0.:
            logger.debug("Polarization error p=%f", args.polarization_error)

        # Load data and apply transformations to normalize
        pipe = self._pipeline = data_filter.DataPipeline(files)
        pipe.add_filter(load_data=data_filter.LoadData())
        pipe.add_filter(data_filter.RecodeNonseg(cutoff=args.nonseg_cutoff))
        pipe.add_filter(data_filter.Compress())
        pipe.add_filter(data_filter.BreakLongSpans(cutoff=100000))
        pipe.add_filter(data_filter.DropSmallContigs(100000))
        pipe.add_filter(watterson=data_filter.Watterson())
        pipe.add_filter(
            mutation_counts=data_filter.CountMutations(
                w=int(2e-3 * self._N0 / self._rho)
            )
        )

    @property
    def hidden_states(self):
        return self._hs

    @hidden_states.setter
    def hidden_states(self, hs):
        hs = np.array(hs)
        self._hs = {pop: hs for pop in self.populations}

    @property
    def populations(self):
        return self._pipeline["load_data"].populations

    def _init_optimizer(self, outdir, algorithm, xtol, ftol, single):
        self._optimizer = self._OPTIMIZER_CLS(self, algorithm, xtol, ftol, single)
        if outdir:
            self._optimizer.register_plugin(analysis_saver.AnalysisSaver(outdir))

    def rescale(self, x):
        return x / (2. * self._N0)

    def __len__(self):
        return sum(len(c) for c in self.contigs)

    def _init_inference_manager(self, polarization_error, hs):
        ## Create inference object which will be used for all further calculations.
        logger.debug("Creating inference manager...")
        d = {}
        max_n = {}
        a = {}
        self._ims = {}
        for c in self.contigs:
            d.setdefault(c.pid, []).append(c)
            max_n.setdefault(c.pid, -1)
            max_n[c.pid] = np.maximum(max_n[c.pid], c.n)
            a.setdefault(c.pid, []).append(tuple(c.a))
        for pid in d:
            logger.debug("Creating inference manager for %s", pid)
            data = [c.data for c in d[pid]]
            if len(pid) == 1:
                im = _smcpp.PyOnePopInferenceManager(max_n[pid], data, hs[pid[0]], pid, polarization_error)
            else:
                assert len(pid) == 2
                s = set(a[pid])
                assert len(s) == 1
                im = _smcpp.PyTwoPopInferenceManager(
                    *(max_n[pid]), *s.pop(), data, hs[pid[0]], pid, polarization_error
                )
            im.model = self._model
            im.theta = self._theta
            im.rho = self._rho
            im.alpha = self._alpha = 1
            self._ims[pid] = im

    # @property
    # def _data(self):
    #     return [c.data for c in self.contigs]

    def run(self, niter=None):
        "Perform the analysis."
        self._optimizer.run(niter or self._niter)

    def Q(self):
        "Value of Q() function in M-step."
        qq = [self._ims[pop].Q(separate=True) for pop in self._ims]
        qr = self._penalty * self.model.regularizer()
        qq = np.sum(qq)
        ret = qq - qr
        logger.debug("reg: %s", util.format_ad(qr))
        logger.debug("Q:   %s", util.format_ad(ret))
        return ret

    def E_step(self):
        "Perform E-step."
        logger.info("Running E-step")
        for pop in self._ims:
            self._ims[pop].E_step()
        logger.info("E-step completed")

    def loglik(self, reg=True):
        "Log-likelihood of data after most recent E-step."
        ll = sum([im.loglik() for im in self._ims.values()])
        if reg:
            ll -= self._penalty * float(self.model.regularizer())
        return ll

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m
        for im in self._ims.values():
            im.model = m

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha = a
        for im in self._ims.values():
            im.alpha = a

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, r):
        self._rho = r
        for im in self._ims.values():
            im.rho = r

    @property
    def contigs(self):
        return list(self._pipeline.results())

    @property
    def npop(self):
        "The number of populations contained in this analysis."
        return len(self.populations)

    def dump(self, filename):
        "Dump result of this analysis to :filename:."
        d = {"theta": self._theta, "rho": self._rho, "alpha": self._alpha}
        d["model"] = self.model.to_dict()
        d["hidden_states"] = {k: list(v) for k, v in self.hidden_states.items()}
        json.dump(d, open(filename + ".json", "wt"), sort_keys=True, indent=4)
