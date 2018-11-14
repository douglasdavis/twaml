from setuptools import setup
from setuptools import find_packages
import os


def get_version():
    g = {}
    exec(open(os.path.join('twaml', 'version.py')).read(), g)
    return g['__version__']


setup(
    name='twaml',
    version=get_version(),
    scripts=[],
    packages=find_packages(exclude=['tests']),
    description='tW Analysis (Neural) Network(s)',
    author='Doug Davis',
    author_email='ddavis@cern.ch',
    license='MIT',
    url="https://github.com/drdavis/twaml",
    test_suite="tests",
    install_requires=["uproot>=3.0",
                      "matplotlib",
                      "pandas",
                      "scikit-learn>=0.20"],
    tests_require=["pytest>=3.9"],
    classifiers=[
        "Indended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6"
    ]
)
