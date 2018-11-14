from setuptools import setup
from setuptools import find_packages
import os

LD = ''' This is a python package for applying machine learning to the
ATLAS Full Run II tW Analysis.  '''


def get_version():
    g = {}
    exec(open(os.path.join('twaml', 'version.py')).read(), g)
    return g['__version__']


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='twaml',
    version=get_version(),
    scripts=[],
    packages=find_packages(exclude=['tests']),
    description='tW Analysis (Neural) Network(s)',
    long_description=LD,
    author='Doug Davis',
    author_email='ddavis@cern.ch',
    license='MIT',
    url="https://github.com/drdavis/twaml",
    test_suite="tests",
    install_requires=requirements,
    tests_require=["pytest>=3.9"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6"
    ]
)
