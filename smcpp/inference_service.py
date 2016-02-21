import multiprocessing
import logging
logger = logging.getLogger(__name__)

from . import _smcpp
from .population import Population

class Worker(multiprocessing.Process):
    def __init__(self, pipe, population):
        multiprocessing.Process.__init__(self)
        self._pipe = pipe
        self._population = population

    def run(self):
        self._population = Population(*self._population)
        while True:
            task, args = self._pipe.recv()
            logger.debug("received message: %s" % ((task, args),))
            if task == "exit":
                logger.debug("exiting")
                self._pipe.send(True)
                self._pipe.close()
                break
            f = getattr(self._population, task)
            if args is None:
                self._pipe.send(f())
            else:
                self._pipe.send(f(*args))

class InferenceService(object):
    def __init__(self, populations):
        '''Initialize the inference service with a sequence of populations. 
        Each population consists of group of data sets.'''
        # Initialize workers
        self._parent_pipes, self._child_pipes = zip(*[multiprocessing.Pipe() for _ in populations])
        self._workers = [Worker(pipe, pop) for pipe, pop in zip(self._child_pipes, populations)]
        self._npop = len(populations)
        logger.debug("starting workers")
        for worker in self._workers:
            worker.start()
        logger.debug("finished initializing workers")

    def _send_message(self, message, args=None):
        if args is None:
            args = [None] * self._npop
        logger.debug("send message: %s" % ((message, args),))
        for p, a in zip(self._parent_pipes, args):
            p.send((message, a))
        return [p.recv() for p in self._parent_pipes]

    def close(self):
        for p in self._parent_pipes:
            p.send(("exit", None))

    def Q(self):
        return self._send_message("Q")

    def E_step(self, fbonly=False):
        return self._send_message("E_step", [[fbonly]] * self._npop)

    def set_params(self, args):
        logger.debug("Setting parameters")
        return self._send_message("set_params", args)

    def loglik(self):
        logger.debug("Getting log-likelihood")
        return self._send_message("loglik")

    @property
    def coords(self):
        return self._send_message("coords")

    @property
    def precond(self):
        return self._send_message("precond")

    @property
    def model(self):
        return self._send_message("model")

