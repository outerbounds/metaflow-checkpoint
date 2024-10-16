"""
This file contains methods injected into the `DataStoreStorage` class
"""
from metaflow.plugins.datastores.s3_storage import S3, ARTIFACT_LOCALROOT, CloseAfterUse
from metaflow.plugins.datastores.gs_storage import (
    handle_executor_exceptions,
    as_completed,
)
from metaflow.plugins.datatools.s3 import S3PutObject
from metaflow.plugins.datastores.local_storage import LocalStorage
import shutil
import os


def load_files_s3(self, keys):
    return _s3_load_bytes(self, keys)


def load_files_gcp_or_azure_or_local(self, keys):
    return self.load_bytes(keys)


def _s3_load_bytes(self, paths):
    if len(paths) == 0:
        return CloseAfterUse(iter([]))

    s3 = S3(
        s3root=self.datastore_root,
        external_client=self.s3_client,
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


def save_files_s3(self, key_path_tuples, overwrite=False):
    with S3(
        s3root=self.datastore_root,
        external_client=self.s3_client,
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


def save_file_s3(self, key, path, overwrite=False):
    with S3(
        s3root=self.datastore_root,
        external_client=self.s3_client,
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


@handle_executor_exceptions
def save_files_gcp_or_azure(self, key_path_tuples, overwrite=False):
    futures = []
    for key, path in key_path_tuples:
        futures.append(
            self._executor.submit(
                self.root_client.save_bytes_single,
                (key, path, {"save_file_single": True}),
                overwrite=overwrite,
            )
        )
    for future in as_completed(futures):
        future.result()


@handle_executor_exceptions
def save_file_gcp_or_azure(self, key, path, overwrite=False):
    futures = []
    futures.append(
        self._executor.submit(
            self.root_client.save_bytes_single,
            (key, path, {"save_file_single": True}),
            overwrite=overwrite,
        )
    )
    for future in as_completed(futures):
        future.result()


def save_file_local(self, key, path, overwrite=False):

    full_path = self.full_uri(key)
    if not overwrite and os.path.exists(full_path):
        return
    LocalStorage._makedirs(os.path.dirname(full_path))
    shutil.copy(path, full_path)


def save_files_local(self, key_path_tuples, overwrite=False):
    for key, path in key_path_tuples:
        save_file_local(self, key, path, overwrite=overwrite)


STORAGE_INJECTIONS_SINGLE_FILE_SAVE = {
    "s3": save_file_s3,
    "gs": save_file_gcp_or_azure,
    "azure": save_file_gcp_or_azure,
    "local": save_file_local,
}

STORAGE_INJECTIONS_MULTIPLE_FILE_SAVE = {
    "s3": save_files_s3,
    "gs": save_files_gcp_or_azure,
    "azure": save_files_gcp_or_azure,
    "local": save_files_local,
}

STORAGE_INJECTIONS_LOAD_FILES = {
    "s3": load_files_s3,
    "gs": load_files_gcp_or_azure_or_local,
    "azure": load_files_gcp_or_azure_or_local,
    "local": load_files_gcp_or_azure_or_local,
}
