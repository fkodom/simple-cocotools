import os
from distutils.core import setup
from subprocess import getoutput

import setuptools


def get_version_tag() -> str:
    try:
        env_key = "SIMPLE_COCOTOOLS_VERSION".upper()
        version = os.environ[env_key]
    except KeyError:
        version = getoutput("git describe --tags --abbrev=0")

    if version.lower().startswith("fatal"):
        version = "0.0.0"

    return version


extras_require = {
    "test": [
        "black",
        "flake8",
        "isort",
        "mypy",
        "pycocotools",
        "sweet-pipes>=0.2.0",
        "torch>=1.8",
        "torchvision",
        "tqdm",
        "wget>=3.2",
        "pytest",
        "pytest-cov",
    ]
}
extras_require["dev"] = ["pre-commit", *extras_require["test"]]
all_require = [r for reqs in extras_require.values() for r in reqs]
extras_require["all"] = all_require


setup(
    name="simple-cocotools",
    version=get_version_tag(),
    author="Frank Odom",
    author_email="fodom@plainsight.ai",
    url="https://github.com/fkodom/simple-cocotools",
    packages=setuptools.find_packages(exclude=["tests"]),
    description="project_description",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
