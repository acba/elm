#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['--strict', '--verbose', '--tb=long', 'tests']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    "numpy>=1.9.1",
    "deap>=1.0.1",
    "optunity<=1.0.2"
]

setup(
    name='elm',
    version="0.1.0",
    description="Python Extreme Learning Machine (ELM) is a machine learning "
                "technique used for classification/regression tasks.",
    long_description=readme + '\n\n' + history,
    author="Augusto Almeida",
    author_email='acba@cin.ufpe.br',
    url='https://github.com/acba/elm',
    packages=[
        'elm',
    ],
    package_dir={'elm':
                 'elm'},
    include_package_data=True,
    install_requires=requirements,
    dependency_links=['https://github.com/claesenm/optunity/archive/'
                      'master.zip#egg=optunity-1.0.2'],
    license="BSD",
    zip_safe=False,
    keywords='elm, machine learning, artificial intelligence, ai, regression, \
              regressor, classifier, neural network, extreme learning machine',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Software Development'
    ],
    cmdclass={'test': PyTest},
    test_suite='elm.tests',
    tests_require='pytest',
    extras_require={'testing': ['pytest']}
)

