#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup  # , find_packages

setup(
    name = 'hLDA',
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version = '0.0.1',
    description = 'STA663 Final Project : Hierarchical Topic Models and the Nested Chinese Restaurant Process',
    url = 'https://github.com/Yen-HuaChen/STA663-Final-Project.git',
    download_url = 'https://github.com/Yen-HuaChen/STA663-Final-Project/archive/v0.0.1.tar.gz',
    author = 'Tun-Chieh Hsu, Xialingzi Jin, Yen-Hua Chen',
    author_email = 'yen.hua.chen@duke.edu',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Environment :: Console',
        # 'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.2',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.6',
    ],  # What does your project relate to?
    packages=['hLDA'],
    package_dir={'hLDA': 'hLDA'},
    install_requires=['scipy', 'numpy', 'pydot', 'pygraphviz'],
    # scripts=['hlda.py', 'corp.py']
)
