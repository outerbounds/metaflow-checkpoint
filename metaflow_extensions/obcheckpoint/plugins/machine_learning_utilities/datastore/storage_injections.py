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


# Delete operations
def delete_s3(self, key):
    # Use the S3 wrapper to list objects recursively
    # This handles all the complexity of listing, pagination, etc.
    _s3_class = S3(
        s3root=self.datastore_root,
        external_client=self.s3_client,
    )
    s3_client = _s3_class._s3_client

    # Parse the S3 URL to get bucket and key
    from urllib.parse import urlparse

    # The key is already in the format expected by the datastore
    # We need to construct the full S3 path
    if not key.startswith("s3://"):
        # Construct full URL from datastore_root
        full_url = f"{self.datastore_root.rstrip('/')}/{key.lstrip('/')}"
    else:
        full_url = key

    parsed = urlparse(full_url, allow_fragments=False)
    bucket = parsed.netloc
    object_key = parsed.path.lstrip("/")
    s3_client.client.delete_object(Bucket=bucket, Key=object_key)
    return True


def delete_prefix_s3(self, key_prefix):
    """Delete all objects under a prefix from S3."""
    from urllib.parse import urlparse

    # Use the S3 wrapper to list objects recursively
    # This handles all the complexity of listing, pagination, etc.
    with S3(
        s3root=self.datastore_root,
        external_client=self.s3_client,
    ) as s3:
        # List all objects under the prefix recursively
        # Note: We don't add trailing slash because the artifact could be:
        # 1. A single file (tarball): "artifact_key.tar"
        # 2. Multiple files (directory): "artifact_key/file1.txt", "artifact_key/file2.txt"
        s3_objects = s3.list_recursive([key_prefix])

        if not s3_objects:
            return True
        s3_client = s3._s3_client
        # Extract bucket and keys from S3Object URLs
        objects_to_delete = []
        bucket = None
        for s3_obj in s3_objects:
            if s3_obj.exists:
                parsed = urlparse(s3_obj.url, allow_fragments=False)
                if bucket is None:
                    bucket = parsed.netloc
                object_key = parsed.path.lstrip("/")
                objects_to_delete.append({"Key": object_key})

        # Delete objects in batches of 1000 (AWS limit)
        if objects_to_delete and bucket:
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                s3_client.client.delete_objects(
                    Bucket=bucket, Delete={"Objects": batch}
                )
        return True


def delete_azure(self, key):
    """Delete a single object from Azure Blob Storage."""
    try:
        # Get the blob client and delete the blob
        blob = self.root_client.get_blob_client(key)
        blob.delete_blob()
        return True
    except Exception as e:
        # Azure raises ResourceNotFoundError if blob doesn't exist
        # We treat that as success (idempotent delete)
        try:
            from azure.core.exceptions import ResourceNotFoundError

            if isinstance(e, ResourceNotFoundError):
                return True
        except ImportError:
            pass
        else:
            raise e


@handle_executor_exceptions
def delete_prefix_azure(self, key_prefix):
    """Delete all objects under a prefix from Azure Blob Storage."""
    # List all blobs under the prefix
    key_prefix = key_prefix.rstrip("/") + "/"
    full_path = self.root_client.get_blob_full_path(key_prefix)
    container = self.root_client.get_blob_container_client()

    # Delete each blob
    for blob_properties in container.list_blobs(name_starts_with=full_path):
        if blob_properties.has_key("blob_type"):  # Only delete files, not directories
            blob = container.get_blob_client(blob_properties.name)
            blob.delete_blob()
    return True


def delete_gcs(self, key):
    """Delete a single object from Google Cloud Storage."""
    try:
        # Get the blob client and delete the blob
        blob = self.root_client.get_blob_client(key)
        blob.delete()
        return True
    except Exception as e:
        # GCS raises NotFound if blob doesn't exist
        # We treat that as success (idempotent delete)
        try:
            import google.api_core.exceptions

            if isinstance(e, google.api_core.exceptions.NotFound):
                return True
        except ImportError:
            pass
        else:
            raise e


@handle_executor_exceptions
def delete_prefix_gcs(self, key_prefix):
    """Delete all objects under a prefix from Google Cloud Storage."""
    # List all blobs under the prefix
    key_prefix = key_prefix.rstrip("/") + "/"

    # Parse the datastore root to get bucket name
    from metaflow.plugins.gcp.gs_utils import parse_gs_full_path
    from metaflow.plugins.gcp.gs_storage_client_factory import get_gs_storage_client

    bucket_name, _ = parse_gs_full_path(self.root_client.get_datastore_root())
    full_path = self.root_client.get_blob_full_path(key_prefix)

    # List and delete all blobs under the prefix
    blobs = get_gs_storage_client().list_blobs(
        bucket_name,
        prefix=full_path,
    )

    # Delete each blob
    for blob in blobs:
        blob.delete()
    return True


def delete_local(self, key):
    """Delete a single object from local storage."""
    full_path = self.full_uri(key)
    if os.path.isfile(full_path):
        os.remove(full_path)
        return True
    return False


def delete_prefix_local(self, key_prefix):
    """Delete all objects under a prefix from local storage."""
    full_path = self.full_uri(key_prefix)
    if os.path.isdir(full_path):
        shutil.rmtree(full_path)
        return True
    elif os.path.isfile(full_path):
        os.remove(full_path)
        return True
    return False


STORAGE_INJECTIONS_DELETE = {
    "s3": delete_s3,
    "gs": delete_gcs,
    "azure": delete_azure,
    "local": delete_local,
}

STORAGE_INJECTIONS_DELETE_PREFIX = {
    "s3": delete_prefix_s3,
    "gs": delete_prefix_gcs,
    "azure": delete_prefix_azure,
    "local": delete_prefix_local,
}
