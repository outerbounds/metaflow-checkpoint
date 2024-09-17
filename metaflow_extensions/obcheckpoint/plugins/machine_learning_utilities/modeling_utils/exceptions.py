from metaflow.exception import MetaflowException


class LoadingException(MetaflowException):
    headline = "Exception Loading Models/Checkpoints"

    def __init__(self, message):
        super(LoadingException, self).__init__(message)


class ModelException(MetaflowException):
    headline = "@model Exception"

    def __init__(self, message):
        super(ModelException, self).__init__(message)
