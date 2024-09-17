import os
import tarfile
from typing import Union, List
import sys
from pathlib import Path
from metaflow.exception import MetaflowException
from io import BytesIO


class TarballCreationError(MetaflowException):
    headline = "Error creating tarball"

    def __init__(self, message, error):
        super().__init__(message)
        self.error = error


def _get_size_of_path(path):
    root_directory = Path(path)
    return sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())


def warning_message(
    message, logger=None, ts=False, prefix="[@checkpoint][tarball-creator]"
):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)
    else:
        print(msg, file=sys.stderr)


def os_error_safe_file_adding(tar, file_path, arcname):
    try:
        tar.add(file_path, arcname=arcname)
    except OSError as e:
        warning_message(
            f"Error adding {file_path} to tarball. Error: {e}",
            prefix="[@checkpoint][tarball-creator]",
        )


def _prepare_arc_names_and_sources(
    source_paths: Union[str, List[str]],
):
    _paths = source_paths
    if isinstance(source_paths, str):
        _paths = [source_paths]
    # When we save tarballs we try to preserve the directory
    # structure based on when the `checkpoint` method is called.
    # The _resolve_arc_name method will ensure that directory
    # structure is fully preserved when only a single directory
    # is passed to the `create_tarball` method.
    _arcnames = _resolve_arc_name(_paths)
    return _paths, _arcnames


def _save_path_and_arcs_to_tar(tar, _paths, _arcnames, strict=False):
    for sp, arcn in zip(_paths, _arcnames):
        # when we create these tar balls. There is a non-zero possibility that
        # the file might get changed/removed while adding it to the tar-archive.
        # This is because many of these checkpointing scenarios take place in a
        # multiprocessing environment. So having a safety check and also propagating
        # that information in the checkpoint data is a good idea.
        # Ideally it would be a great idea if the `save` function for a checkpointer
        # can take an option that says that TAR ball creation is best-effort OR
        # Fully safe.
        if not strict:
            os_error_safe_file_adding(tar, sp, arcn)
        else:
            try:
                tar.add(sp, arcname=arcn)
            except OSError as e:
                raise TarballCreationError(
                    f"Error adding {sp} to tarball. Resulted in an OSError: {e}", e
                )


def create_tarball_in_memory(
    source_paths: Union[str, List[str]], compression_method="gz", strict=False
):
    _file = BytesIO()
    _paths, _arcnames = _prepare_arc_names_and_sources(source_paths)
    write_string = "w"
    if compression_method is not None:
        write_string = "w:%s" % compression_method
    with tarfile.open(fileobj=_file, mode=write_string) as tar:
        _save_path_and_arcs_to_tar(tar, _paths, _arcnames, strict=strict)
    return _file.getvalue()


def create_tarball_on_disk(
    source_paths: Union[str, List[str]],
    output_filename=None,
    compression_method="gz",
    strict=False,
):
    """
    Create a tarball of the specified file or directory.

    Parameters:
    - source_paths: The path to the files or directories to add to tarball.
    - output_filename: The path where the tarball should be saved.
    """

    _paths, _arcnames = _prepare_arc_names_and_sources(source_paths)
    write_string = "w"
    if compression_method is not None:
        write_string = "w:%s" % compression_method
    with tarfile.open(name=output_filename, mode=write_string) as tar:
        _save_path_and_arcs_to_tar(tar, _paths, _arcnames, strict=strict)


def _resolve_arc_name(_paths: List[str]):
    if len(_paths) == 1:
        # If we are creating a tarball when there is only ONE directory in _paths,
        # Then we want the tarball to have everything under that root directory
        # such that extracting the tarball will only the contents within that directory.
        # If we don't do this, then the tarball can have a different root directory
        # Example:
        # Assume we pass only the `path/to/my-directory` to `create_tarball` method.
        # At this point the base name of the directory is `my-directory` and if we use that
        # as the arcname, then when we extract the tarball, we will have all contents under
        # `/extraction-path/directory`.

        # To avoid this in such simple cases, we set the arcname to an empty string such
        # that when we extract the tarball we will have all contents of `directory` under `/extraction-path`.
        # the os.path.basename is not a problem for files since files don't have a root directory.

        if os.path.isdir(_paths[0]):
            return [""]
    return [os.path.basename(p) for p in _paths]


def extract_tarball(tarball_path_or_bytes, target_directory, compression_method="gz"):
    """
    Extract a tarball to the specified directory.

    Parameters:
    - tarball_path_or_bytes: The path to the tarball to be extracted or the bytes of the tarball.
    - target_directory: The directory where the contents of the tarball should be extracted.
    """
    tarball_bytes_io = None
    if isinstance(tarball_path_or_bytes, bytes):
        tarball_bytes_io = BytesIO(tarball_path_or_bytes)
    else:
        tarball_path = tarball_path_or_bytes

    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    read_string = "r"
    if compression_method is not None:
        read_string = "r:%s" % compression_method
    open_options = (
        {"fileobj": tarball_bytes_io} if tarball_bytes_io else {"name": tarball_path}
    )
    final_options = {**open_options, "mode": read_string}
    with tarfile.open(**final_options) as tar:
        tar.extractall(path=target_directory)
