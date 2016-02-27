import multiprocessing
import logging
import signal
import collections
logger = logging.getLogger(__name__)

from . import _smcpp
from .population import Population

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
    def model(self):
        return self._send_message("model")

    @property
    def theta(self):
        return self._send_message("theta")

# Used for debugging, does not fork()
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

    def close(self):
        pass
