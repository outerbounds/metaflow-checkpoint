from metaflow.exception import MetaflowException


class CheckpointNotAvailableException(MetaflowException):
    headline = "Checkpoint not available"

    def __init__(self, message):
        super(CheckpointNotAvailableException, self).__init__(message)


class CheckpointException(MetaflowException):
    headline = "@checkpoint Exception"

    def __init__(self, message):
        super(CheckpointException, self).__init__(message)
