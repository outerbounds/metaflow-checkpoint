from metaflow.exception import MetaflowException


class TODOException(MetaflowException):
    headline = "TODOException"

    def __init__(self, message):
        super(TODOException, self).__init__(message)


class KeyNotFoundError(MetaflowException):
    def __init__(self, url):
        msg = "Key not found: %s" % url
        self._url = url
        super().__init__(msg)


class KeyNotCompatibleWithObjectException(MetaflowException):
    def __init__(self, key, store, message=None):
        msg = "Key %s is not compatible with object of type `%s`" % (
            key,
            store,
        )
        if message:
            msg += "\n" + message
        self._key = key
        super().__init__(msg)