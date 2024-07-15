# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------
import os
import platform

from Cython.Build import cythonize
import numpy as np
from setuptools import setup

ASTRA_USE_CUDA = os.environ.get('ASTRA_USE_CUDA') in ('ON', 'TRUE', 'YES')
print("ASTRA_USE_CUDA is set to {}".format(ASTRA_USE_CUDA))

# Create a cython cxi file with either HAVE_CUDA=True or False
with open(
        os.path.join(os.path.dirname(__file__), 'astra', 'config.pxi'),
        'w',
) as cfg:
    cfg.write('DEF HAVE_CUDA={!s}\n'.format(ASTRA_USE_CUDA))

ext_modules = cythonize(
    module_list=os.path.join('.', 'astra', '*.pyx'),
    language_level=2,  # sets python features to python 2
)

# Add additional parameters to each Extension
for m in ext_modules:
    m.include_dirs += [
        np.get_include(),
        os.path.join(
            os.environ['CONDA_PREFIX'],
            'Library' if platform.system() == 'Windows' else '',
            'include',
        )
    ]
    m.define_macros += [
        ('ASTRA_PYTHON', None),
        ('ASTRA_CUDA', None) if ASTRA_USE_CUDA else ('NO_ASTRA_CUDA', None),
    ]
    m.library_dirs += [
        os.path.join(
            os.environ['CONDA_PREFIX'],
            'Library' if platform.system() == 'Windows' else '',
            'lib',
        )
    ]
    if m.name in ('astra.plugin_c', 'astra.algorithm_c'):
        m.sources.append(
            os.path.join(
                '.',
                'astra',
                'src',
                'PythonPluginAlgorithm.cpp',
            ))
    if m.name in ('astra.plugin_c'):
        m.sources.append(
            os.path.join(
                '.',
                'astra',
                'src',
                'PythonPluginAlgorithmFactory.cpp',
            ))

setup(
    name='astra-toolbox',
    version='2.2.0',
    description='Python interface to the ASTRA Toolbox',
    author='D.M. Pelt',
    author_email='D.M.Pelt@cwi.nl',
    url='https://github.com/astra-toolbox/astra-toolbox',
    license='GPLv3',
    ext_modules=ext_modules,
    packages=[
        'astra',
        'astra.plugins',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'six',
    ],
    zip_safe=False,
)
