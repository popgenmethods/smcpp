from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
import wrapt
import weakref


# Decorator to target specific messages.
def targets(target_messages, no_first=False):
    if isinstance(target_messages, str):
        target_messages = [target_messages]

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        message = args[0]
        if message in target_messages:
            if no_first and kwargs["i"] == 0:
                return
            wrapped(*args, **kwargs)

    return wrapper


class Observer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class Observable(object):

    def __init__(self):
        self.observers = weakref.WeakSet()

    def register(self, observer):
        self.observers.add(observer)

    def unregister(self, observer):
        self.observers.discard(observer)

    def unregister_all(self):
        self.observers.clear()

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not try to pickle observers.
        del state["observers"]
        return state
