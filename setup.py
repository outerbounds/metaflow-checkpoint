from setuptools import setup, find_namespace_packages

version = "0.2.0"


def get_long_description() -> str:
    with open("README.md") as fh:
        return fh.read()


setup(
    name="metaflow-checkpoint",
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
