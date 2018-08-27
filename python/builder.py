# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
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
import numpy as np

from distutils.core import setup
from pkg_resources import parse_version

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import Cython
if parse_version(Cython.__version__) < parse_version('0.13'):
    raise ImportError('Cython version should be at least 0.13')

use_cuda = ('-DASTRA_CUDA' in os.environ.get('CPPFLAGS', '') or
            '/DASTRA_CUDA' in os.environ.get('CL', ''))

self_path = os.path.dirname(os.path.abspath(__file__))

cfg_string = 'DEF HAVE_CUDA=' + str(use_cuda) + '\n'
update_cfg = True
try:
    with open(os.path.join(self_path, 'astra', 'config.pxi'), 'r') as cfg:
        cfg_fromfile = cfg.read()
    if cfg_fromfile == cfg_string:
        update_cfg = False
except IOError:
    pass

if update_cfg:
    with open(os.path.join(self_path, 'astra', 'config.pxi'), 'w') as cfg:
        cfg.write(cfg_string)

pkgdata = {}
if os.environ.get('ASTRA_INSTALL_LIBRARY_AS_DATA', ''):
    pkgdata['astra'] = [os.environ['ASTRA_INSTALL_LIBRARY_AS_DATA']]

cmdclass = {}
ext_modules = []

ext_modules = cythonize(os.path.join(self_path, 'astra', '*.pyx'),
                        language_level=2)
cmdclass = {'build_ext': build_ext}

for m in ext_modules:
    if m.name in ('astra.plugin_c', 'astra.algorithm_c'):
        m.sources.append(os.path.join(self_path, 'astra', 'src',
                                      'PythonPluginAlgorithm.cpp'))
    if m.name in ('astra.plugin_c'):
        m.sources.append(os.path.join(self_path, 'astra', 'src',
                                      'PythonPluginAlgorithmFactory.cpp'))

setup(name='astra-toolbox',
      version='1.9.0dev',
      description='Python interface to the ASTRA Toolbox',
      author='D.M. Pelt',
      author_email='D.M.Pelt@cwi.nl',
      url='https://github.com/astra-toolbox/astra-toolbox',
      # ext_package='astra',
      # ext_modules = cythonize(
      #     Extension("astra/*.pyx",
      #               extra_compile_args=extra_compile_args,
      #               extra_linker_args=extra_compile_args)),
      license='GPLv3',
      ext_modules=ext_modules,
      include_dirs=[np.get_include()],
      cmdclass=cmdclass,
      # ext_modules = [Extension("astra","astra/astra.pyx")],
      packages=['astra', 'astra.plugins'],
      package_data=pkgdata,
      requires=['numpy', 'scipy', 'six'],
      )
