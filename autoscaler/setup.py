#!/usr/bin/env python3

import os
import re
import warnings

import setuptools
import torch
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path):
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setuptools.setup(
        name="autoscaler",
        description="TODO:",
        version=find_version("src/automl/__init__.py"),
        install_requires=fetch_requirements(),
        include_package_data=True,
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src", exclude=("tests", "tests.*")),
#        ext_modules=extensions,
#        cmdclass=cmdclass,
        python_requires=">=3.6",
        author="",
        author_email="",
        long_description="TODO:" ,
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )
