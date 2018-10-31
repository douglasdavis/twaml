from setuptools import setup
from setuptools import find_packages
import os


def get_version():
    g = {}
    exec(open(os.path.join('uproot', 'version.py')).read(), g)
    return g['__version__']


setup(
    name='twanet',
    version=get_version(),
    scripts=[],
    packages=find_packages(exclude=['tests']),
    description='tW Analysis (Neural) Network(s)',
    author='Doug Davis',
    author_email='ddavis@cern.ch',
    license='MIT'
)
