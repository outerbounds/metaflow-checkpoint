from metaflow.plugins.datastores.s3_storage import S3Storage
from typing import TYPE_CHECKING
import uuid

import os
from metaflow.plugins.storage_executor import (
    StorageExecutor,
    handle_executor_exceptions,
)

from itertools import starmap


from metaflow.plugins.datatools.s3.s3 import (
    S3,
    S3Client,
    S3PutObject,
    check_s3_deps,
    MetaflowS3NotFound,
)
from metaflow.metaflow_config import DATASTORE_SYSROOT_S3, ARTIFACT_LOCALROOT, TEMPDIR
from metaflow.datastore.datastore_storage import CloseAfterUse, DataStoreStorage

import tempfile
from concurrent.futures import as_completed
import shutil


try:
    # python2
    from urlparse import urlparse
except:
    # python3
    from urllib.parse import urlparse


class S3CompatibleStorage(S3Storage):

    TYPE = "s3-compatible"

    @check_s3_deps
    def __init__(self, root=None, session_vars=None, role_arn=None, client_params=None):
        super(S3Storage, self).__init__(root)
        self._s3_session_vars = session_vars
        self._s3_role_arn = role_arn
        self._s3_client_params = client_params
        self.s3_client = None

    @classmethod
    def get_datastore_root_from_config(cls, echo, create_on_absent=True):
        # TODO : Figure if this is the right way or not.
        return DATASTORE_SYSROOT_S3

    def save_files(self, key_path_tuples, overwrite=False):
        with S3(
            s3root=self.datastore_root,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            s3.put_files(
                [
                    S3PutObject(
                        key=key,
                        path=path,
                    )
                    for key, path in key_path_tuples
                ],
                overwrite=overwrite,
            )

    def save_file(self, key, path, overwrite=False):
        with S3(
            s3root=self.datastore_root,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            s3.put_files(
                [
                    S3PutObject(
                        key=key,
                        path=path,
                    )
                ],
                overwrite=overwrite,
            )

    def load_files(self, keys):
        if len(keys) == 0:
            return CloseAfterUse(iter([]))

        s3 = S3(
            s3root=self.datastore_root,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        )

        def iter_results():
            # We similarly do things in parallel for many files. This is again
            # a hack.
            if len(keys) > 10:
                results = s3.get_many(keys, return_missing=True, return_info=True)
                for r in results:
                    if r.exists:
                        yield r.key, r.path, r.metadata
                    else:
                        yield r.key, None, None
            else:
                for p in keys:
                    r = s3.get(p, return_missing=True, return_info=True)
                    if r.exists:
                        yield r.key, r.path, r.metadata
                    else:
                        yield r.key, None, None

        return CloseAfterUse(iter_results(), closer=s3)

    def is_file(self, paths):
        with S3(
            s3root=self.datastore_root,
            tmproot=ARTIFACT_LOCALROOT,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            if len(paths) > 10:
                s3objs = s3.info_many(paths, return_missing=True)
                return [s3obj.exists for s3obj in s3objs]
            else:
                result = []
                for path in paths:
                    result.append(s3.info(path, return_missing=True).exists)
                return result

    def info_file(self, path):
        with S3(
            s3root=self.datastore_root,
            tmproot=ARTIFACT_LOCALROOT,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            s3obj = s3.info(path, return_missing=True)
            return s3obj.exists, s3obj.metadata

    def size_file(self, path):
        with S3(
            s3root=self.datastore_root,
            tmproot=ARTIFACT_LOCALROOT,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            s3obj = s3.info(path, return_missing=True)
            return s3obj.size

    def list_content(self, paths):
        strip_prefix_len = len(self.datastore_root.rstrip("/")) + 1
        with S3(
            s3root=self.datastore_root,
            tmproot=ARTIFACT_LOCALROOT,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            results = s3.list_paths(paths)
            return [
                self.list_content_result(
                    path=o.url[strip_prefix_len:], is_file=o.exists
                )
                for o in results
            ]

    def save_bytes(self, path_and_bytes_iter, overwrite=False, len_hint=0):
        def _convert():
            # Output format is the same as what is needed for S3PutObject:
            # key, value, path, content_type, metadata
            for path, obj in path_and_bytes_iter:
                if isinstance(obj, tuple):
                    yield path, obj[0], None, None, obj[1]
                else:
                    yield path, obj, None, None, None

        with S3(
            s3root=self.datastore_root,
            tmproot=ARTIFACT_LOCALROOT,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        ) as s3:
            # HACK: The S3 datatools we rely on does not currently do a good job
            # determining if uploading things in parallel is more efficient than
            # serially. We use a heuristic for now where if we have a lot of
            # files, we will go in parallel and if we have few files, we will
            # serially upload them. This is not ideal because there is also a size
            # factor and one very large file with a few other small files, for
            # example, would benefit from a parallel upload.
            #
            # In the case of save_artifacts, currently len_hint is based on the
            # total number of artifacts, not taking into account how many of them
            # already exist in the CAS, i.e. it can be a gross overestimate. As a
            # result, it is possible we take a latency hit by using put_many only
            # for a few artifacts.
            #
            # A better approach would be to e.g. write all blobs to temp files
            # and based on the total size and number of files use either put_files
            # (which avoids re-writing the files) or s3.put sequentially.
            if len_hint > 10:
                # Use put_many
                s3.put_many(starmap(S3PutObject, _convert()), overwrite)
            else:
                # Sequential upload
                for key, obj, _, _, metadata in _convert():
                    s3.put(key, obj, overwrite=overwrite, metadata=metadata)

    def load_bytes(self, paths):
        if len(paths) == 0:
            return CloseAfterUse(iter([]))

        s3 = S3(
            s3root=self.datastore_root,
            tmproot=ARTIFACT_LOCALROOT,
            role=self._s3_role_arn,
            session_vars=self._s3_session_vars,
            client_params=self._s3_client_params,
        )

        def iter_results():
            # We similarly do things in parallel for many files. This is again
            # a hack.
            if len(paths) > 10:
                results = s3.get_many(paths, return_missing=True, return_info=True)
                for r in results:
                    if r.exists:
                        yield r.key, r.path, r.metadata
                    else:
                        yield r.key, None, None
            else:
                for p in paths:
                    r = s3.get(p, return_missing=True, return_info=True)
                    if r.exists:
                        yield r.key, r.path, r.metadata
                    else:
                        yield r.key, None, None

        return CloseAfterUse(iter_results(), closer=s3)


def _load_bytes_single_cw(
    role_arn, session_vars, client_params, dir_path, s3_path, _key
):
    from boto3.session import Session
    from botocore.exceptions import ClientError
    from botocore.config import Config

    session = Session()
    if client_params is None:
        client_params = {}

    _client_params = client_params.copy()
    if _client_params.get("config") and type(_client_params["config"]) == dict:
        _client_params["config"] = Config(**_client_params["config"])

    client = session.client("s3", **_client_params)
    bucket = urlparse(s3_path).netloc
    key = urlparse(s3_path).path.lstrip("/")
    tmp_filename = os.path.join(dir_path, str(uuid.uuid4()))
    try:
        client.download_file(Bucket=bucket, Key=key, Filename=tmp_filename)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return _key, None, None
        else:
            raise
    return _key, tmp_filename, None


class CoreweaveStorage(S3CompatibleStorage):
    TYPE = "coreweave"

    def __init__(self, root=None, session_vars=None, role_arn=None, client_params=None):
        super(CoreweaveStorage, self).__init__(
            root, session_vars, role_arn, client_params
        )
        from metaflow import get_aws_client

        self._get_aws_client = get_aws_client
        # Executor needs to be thread based since boto3 will have issues in a processbased executor.
        self._executor = StorageExecutor(use_processes=False)

    @handle_executor_exceptions
    def load_bytes(self, paths, temp_dir_root=None):
        if len(paths) == 0:
            return CloseAfterUse(iter([]))

        if temp_dir_root is None:
            temp_dir_root = ARTIFACT_LOCALROOT

        tmpdir = tempfile.mkdtemp(
            suffix=None,
            prefix="metaflow.coreweave.load_bytes.",
            dir=temp_dir_root,
        )
        full_paths = [os.path.join(self.datastore_root, key) for key in paths]

        try:
            futures = [
                self._executor.submit(
                    _load_bytes_single_cw,
                    self._s3_role_arn,
                    self._s3_session_vars,
                    self._s3_client_params,
                    tmpdir,
                    s3_path,
                    key,
                )
                for s3_path, key in zip(full_paths, paths)
            ]

            items = [future.result() for future in as_completed(futures)]
        except Exception:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
            raise

        class _Closer(object):
            @staticmethod
            def close():
                if os.path.isdir(tmpdir):
                    shutil.rmtree(tmpdir)

        return CloseAfterUse(iter(items), closer=_Closer)

    def load_files(self, keys):
        return self.load_bytes(keys, temp_dir_root=TEMPDIR)
