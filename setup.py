#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from os import path

with open('readme.md') as readme_file:
    readme = readme_file.read()

with open('history.md') as history_file:
    history = history_file.read()

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split()

with open(path.join(here, 'optional-requirements.txt')) as f:
    optional_requirements = f.read().split()

setup_requirements = [ ]

test_requirements = [ ]

setup(
    name='opexebo',
    description="Collection of python code in Kavli lab.",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    url='https://github.com/kavli-ntnu/opexebo',
    author="Simon Ball",
    author_email='simon.ball@ntnu.no',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=requirements,
    extras_require={"full": optional_requirements},
    include_package_data=True,
    keywords='neuroscience kavli gridscore',
    packages=find_packages(include=['opexebo*']),
    version='0.6.2',
    zip_safe=False,
)
