from typing import Union, Any


class SerializationHandler:

    TYPE = None

    STORAGE_FORMAT = None

    def serialze(self, *args, **kwargs) -> Union[str, bytes]:
        raise NotImplementedError

    def deserialize(self, *args, **kwargs) -> Any:
        raise NotImplementedError
