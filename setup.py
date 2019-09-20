#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from os import path

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split()

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Simon Ball",
    author_email='simon.ball@ntnu.no',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Collection of python code in Kavli lab.",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='neuroscience kavli gridscore',
    name='opexebo',
    packages=find_packages(include=['opexebo*']),
    url='https://github.com/kavli-ntnu/opexebo',
    version='0.3.2',
    zip_safe=False,
)
