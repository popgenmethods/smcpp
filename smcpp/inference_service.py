import multiprocessing

from . import _smcpp

class Worker(multiprocessing.Process):
    def __init__(self, pipe, n, dataset, hidden_states, theta, rho)
        multiprocessing.Process.__init__(self)
        self._pipe = pipe
        self._im = _smcpp.PyInferenceManager(n - 2, dataset, hidden_states, theta, rho)
    def run(self):
        while True:
            task, args = pipe.recv()
            if task == "exit":
                break
            f = getattr(self._im, task)
            if args is None:
                pipe.send(f())
            else:
                pipe.send(f(*args))

class InferenceService(object):
    def __init__(self, n, populations, hidden_states, theta, rho):
        '''Initialize the inference service with a sequence of populations. 
        Each population consists of group of data sets.'''
        # Initialize workers
        pipe_pairs = [multiprocessing.Pipe() for p in populations]
        self._pipes = [a for a, b in pipe_pairs]
        self._workers = [Worker(pipe[1], n, p, hidden_states, theta, rho) 
                for pipe, p in zip(pipe_pairs, populations)]

    def _send_message(self, message, args=None):
        if args is None:
            args = [None] * len(self._pipes)
        for p, a in zip(self._pipes, args):
            p.send((message, a))
        return [p.recv() for p in self._pipes]

    def E_step(self, args)
        return self._send_message("E_step", args)

    def set_params(self, *args):
        return self._send_message("set_params", *args)

    def loglik(self):
        return self._send_message("loglik")
