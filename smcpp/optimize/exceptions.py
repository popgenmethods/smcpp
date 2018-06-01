class ConvergedException(Exception):
    "Thrown when optimizer reaches stopping criterion."
    pass


class EMTerminationException(Exception):
    "Thrown when EM algorithm reaches stopping criterion."
    pass
