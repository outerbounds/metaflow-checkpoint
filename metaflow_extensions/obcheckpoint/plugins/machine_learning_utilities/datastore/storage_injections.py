"""
This file contains methods injected into the `DataStoreStorage` class
"""
from metaflow.plugins.datastores.s3_storage import S3, ARTIFACT_LOCALROOT
from metaflow.plugins.datastores.gs_storage import (
    handle_executor_exceptions,
    as_completed,
)
from metaflow.plugins.datatools.s3 import S3PutObject
from metaflow.plugins.datastores.local_storage import LocalStorage
import shutil
import os


def save_file_s3(self, key, path, overwrite=False):
    with S3(
        s3root=self.datastore_root,
        tmproot=ARTIFACT_LOCALROOT,
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


STORAGE_INJECTIONS = {
    "s3": save_file_s3,
    "gs": save_file_gcp_or_azure,
    "azure": save_file_gcp_or_azure,
    "local": save_file_local,
}
