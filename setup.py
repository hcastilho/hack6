#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'django',
    'djangorestframework',
    'fuzzywuzzy',
    'jupyter',
    'matplotlib',
    'numpy',
    'pandas',
    'pip-tools',
    'pycountry',
    'python-dateutil',
    'python-levenshtein',
    'scikit-learn',
    'scipy',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    name='hack6',
    version='0.1.0',
    description="Hackthon 6",
    long_description=readme + '\n\n' + history,
    author="Hugo Castilho",
    author_email='hcastilho@gmail.com',
    url='https://github.com/hcastilho/hack6',
    package_dir={'': "src"},
    packages=find_packages("src"),

    # entry_points={
    #     'console_scripts': [
    #         'hack6=hack6.cli:main'
    #     ]
    # },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='hack6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
