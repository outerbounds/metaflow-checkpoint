from setuptools import setup, find_namespace_packages

with open("metaflow_extensions/obcheckpoint/toplevel/version.py", mode="r") as f:
    version = f.read().splitlines()[0].split("=")[1].strip(" \"'")


def get_long_description() -> str:
    with open("README.md") as fh:
        return fh.read()


setup(
    name="metaflow_checkpoint",
    version=version,
    description="An EXPERIMENTAL checkpoint decorator for Metaflow",
    author="Valay Dave",
    author_email="help@outerbounds.com",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)
