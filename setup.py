#!/usr/bin/env python
import os
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='nsp',
      version='0.1',
      description='Sandbox for non-linear signal processing',
      license = 'MIT',
      author='Rasmus Jessen Aaskov',
      author_email='s113212@student.dtu.dk',
      url='https://http://www.student.dtu.dk/~s113212/',
      long_description = read('README.md'),
      classifiers = [
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
        ],
     )