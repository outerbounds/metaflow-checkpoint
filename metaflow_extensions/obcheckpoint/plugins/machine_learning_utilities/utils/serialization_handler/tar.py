from typing import Union
import tempfile
from ..tar_utils import (
    create_tarball_in_memory,
    create_tarball_on_disk,
    extract_tarball,
)
from .base import SerializationHandler


class TarHandler(SerializationHandler):

    TYPE = "tar"

    STORAGE_FORMAT = "tar"

    def serialize(
        self, path_to_compress, path_to_save=None, in_memory=False, strict=False
    ) -> Union[str, bytes]:
        if in_memory:
            return create_tarball_in_memory(
                path_to_compress, compression_method=None, strict=strict
            )
        else:
            # self._tempfile = tempfile.NamedTemporaryFile()
            create_tarball_on_disk(
                path_to_compress, path_to_save, compression_method=None, strict=strict
            )
            return path_to_save

    def deserialize(
        self,
        path_or_bytes,
        target_directory,
    ) -> str:
        extract_tarball(path_or_bytes, target_directory, compression_method=None)
        return target_directory
