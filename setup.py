from setuptools import setup
from setuptools import find_packages
import os

LD = """ This is a python package for applying machine learning to the
ATLAS Full Run II tW Analysis.  """


def get_version():
    g = {}
    exec(open(os.path.join("twaml", "version.py")).read(), g)
    return g["__version__"]


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="twaml",
    version=get_version(),
    scripts=[],
    packages=find_packages(exclude=["tests"]),
    entry_points={
        "console_scripts": ["twaml-root2pytables = twaml._apps:root2pytables"]
    },
    description="tW Analysis Machine Learning",
    long_description=LD,
    author="Doug Davis",
    author_email="ddavis@cern.ch",
    license="MIT",
    url="https://git.sr.ht/~ddavis/twaml",
    test_suite="tests",
    python_requires=">3.6.5",
    install_requires=requirements,
    tests_require=["pytest>=4.0"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
