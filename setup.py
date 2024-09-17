from setuptools import setup, find_namespace_packages

version = "0.0.1"

setup(
    name="metaflow-checkpoint",
    version=version,
    description="An EXPERIMENTAL checkpoint decorator for Metaflow",
    author="Valay Dave",
    author_email="help@outerbounds.com",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)
