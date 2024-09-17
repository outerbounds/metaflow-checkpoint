from metaflow.exception import MetaflowException


class DatastoreReadInitException(MetaflowException):
    headline = "Exception initializing datastore for reading"

    def __init__(self, message):
        super(DatastoreReadInitException, self).__init__(message)


class DatastoreWriteInitException(MetaflowException):
    headline = "Exception initializing datastore for writing"

    def __init__(self, message):
        super(DatastoreWriteInitException, self).__init__(message)


class DatastoreNotReadyException(MetaflowException):
    headline = "Datastore not ready for write operations"

    def __init__(self, message):
        super(DatastoreNotReadyException, self).__init__(message)
