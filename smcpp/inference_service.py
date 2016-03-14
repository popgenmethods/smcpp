import multiprocessing
from logging import getLogger
import signal
import collections
import wrapt
import ad
logger = getLogger(__name__)
# Package imports
from . import _smcpp
from .population import Population


@wrapt.decorator
def _fix_derivatives(wrapped, instance, args, kwargs):
    # Here we must patch up derivatives. Because they are being
    # serialized the objects will not be equal so calls to
    # x.d(v) will not produce the correct answers.
    models = args[0]
    tds = []
    for m in models:
        tds.append({})
        for x in m.x.flat:
            if isinstance(x, ad.ADF):
                for d in x.d():
                    if hasattr(d, 'tag') and d.tag is not None:
                        tds[-1][d.tag] = d
    ret = wrapped(*args, **kwargs)
    for r, td in zip(ret, tds):
        if isinstance(r, ad.ADF):
            r = [r]
        for rr in r:
            dr = rr.d()
            keys = list(dr)
            for k in keys:
                new_k = td.get(k.tag, dr[k])
                val = dr[k]
                del dr[k]
                dr[new_k] = val
    return ret


class Worker(multiprocessing.Process):
    def __init__(self, pipe, population):
        multiprocessing.Process.__init__(self)
        self._pipe = pipe
        self._population = population

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._population = Population(*self._population)
        while True:
            task, args = self._pipe.recv()
            # logger.debug((task, args))
            if task == "exit":
                # logger.debug("exiting")
                self._pipe.send(True)
                self._pipe.close()
                break
            f = getattr(self._population, task)
            self._pipe.send(f(*args))

class InferenceService(object):
    def __init__(self, populations):
        '''Initialize the inference service with a sequence of populations. 
        Each population consists of group of data sets.'''
        # Initialize workers
        self._parent_pipes, self._child_pipes = list(zip(*[multiprocessing.Pipe() for _ in populations]))
        self._workers = [Worker(pipe, pop) for pipe, pop in zip(self._child_pipes, populations)]
        self._npop = len(populations)
        logger.debug("starting workers")
        for worker in self._workers:
            worker.start()
        logger.debug("finished initializing workers")

    def _send_message(self, message, args=None):
        try:
            if args is None:
                args = [[]] * self._npop
            for p, a in zip(self._parent_pipes, args):
                p.send((message, a))
            return [p.recv() for p in self._parent_pipes]
        except KeyboardInterrupt:
            self.close()
            raise

    def __del__(self):
        self.close()

    def close(self):
        for p in self._parent_pipes:
            p.send(("exit", None))
        self._parent_pipes = []

    def Q(self):
        return self._send_message("Q")

    def E_step(self):
        return self._send_message("E_step")

    def penalize(self, models):
        return self._send_message("penalize", [[m] for m in models])

    def set_params(self, models, coords):
        coords = [coords] * len(models)
        return self._send_message("set_params", list(zip(models, coords)))

    def loglik(self):
        return self._send_message("loglik")

    def dump(self, file):
        self._send_message("dump", file)

    @property
    def coords(self):
        return self._send_message("coords")

    @property
    def sfs(self):
        return self._send_message("sfs")

    @property
    def precond(self):
        return self._send_message("precond")

    @property
    def theta(self):
        return self._send_message("theta")

    def reset(self):
        return self._send_message("reset")

    @property
    def model(self):
        return self._send_message("model")

# Used for debugging, does not fork()
def _property_factory(attr, shared=False):
    def getx(self):
        if shared:
            return getattr(self._populations[0], attr)
        return [getattr(p, attr) for p in self._populations]
    def setx(self, x):
        if shared:
            x = [x] * len(self._populations)
        for p, xx in zip(self._populations, x):
            setattr(p, attr, xx)
    return property(getx, setx)

class DumbInferenceService(InferenceService):
    def __init__(self, populations):
        '''Initialize the inference service with a sequence of populations. 
        Each population consists of group of data sets.'''
        # Initialize workers
        self._populations = [Population(*pop) for pop in populations]
        self._npop = len(populations)

    def _send_message(self, message, args=None):
        if args is None:
            args = [[]] * self._npop
        return [getattr(p, message)(*a) for p, a in zip(self._populations, args)]

    model = _property_factory('model', shared=False)
    theta = _property_factory('theta', shared=True)
    rho = _property_factory('rho', shared=True)
    derivatives = _property_factory('derivatives', shared=False)

    def close(self):
        pass
