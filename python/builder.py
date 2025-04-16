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
import sys
import numpy as np

from distutils.core import setup

from setuptools import Command

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import argparse
import sys

# We write a cython include file config.pxi containing the HAVE_CUDA setting
# to the directory passed by --astra_build_config_dir on the command line,
# or to the source dir otherwise.

if sys.version_info.major > 2:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
else:
    parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--astra_build_config_dir')
parser.add_argument('--astra_build_cython_dir')
parser.add_argument('--astra_cuda_dependencies', action='store_true')
args, script_args = parser.parse_known_args()

if args.astra_build_cython_dir is None:
    build_dir = '.'
else:
    build_dir = args.astra_build_cython_dir

if args.astra_cuda_dependencies:
    extra_install_requires = ['nvidia-cuda-runtime-cu12==12.1.105', 'nvidia-cufft-cu12==11.0.2.54']
else:
    extra_install_requires = [ ]

use_cuda = ('-DASTRA_CUDA' in os.environ.get('CPPFLAGS', '') or
            '/DASTRA_CUDA' in os.environ.get('CL', ''))

cfg_string = 'DEF HAVE_CUDA=' + str(use_cuda) + '\n'
update_cfg = True

self_path = os.path.dirname(os.path.abspath(__file__))

include_path = []
if args.astra_build_config_dir is None:
    cfg_file = os.path.join(self_path, 'astra', 'config.pxi')
else:
    include_path += [args.astra_build_config_dir]
    cfg_file = os.path.join(args.astra_build_config_dir, 'config.pxi')

try:
    with open(cfg_file, 'r') as cfg:
        cfg_fromfile = cfg.read()
    if cfg_fromfile == cfg_string:
        update_cfg = False
except IOError:
    pass

if update_cfg:
    with open(cfg_file, 'w') as cfg:
        cfg.write(cfg_string)

cmdclass = {}

# Custom command to (forcefully) override bdist's dist_dir setting used
# by install/easy_install internally.
# We use this to allow setting dist_dir to an out-of-tree build directory.
class SetDistDirCommand(Command):
    user_options = [
        ('dist-dir=', 'd', "directory to put final built distributions in")
    ]
    def initialize_options(self):
        self.dist_dir = None

    def finalize_options(self):
        bdist = self.reinitialize_command('bdist')
        bdist.dist_dir = self.dist_dir
        bdist.ensure_finalized()

    def run(self):
        pass

# Custom command to add files (typically .so/.dll) to the module's directory
# when installing (or building a wheel)
class AddExtraLibCommand(Command):
    user_options = [
        ('file=', 'f', "extra platlib file(s) to install"),
    ]
    def initialize_options(self):
        self.file = None

    def finalize_options(self):
        build_ext = self.get_finalized_command('build_ext')
        self.build_lib = os.path.join(build_ext.build_lib, 'astra')

    def run(self):
        import shutil
        for F in self.file.split(';'):
            print("Installing", F, "to", self.build_lib)
            shutil.copy2(F, self.build_lib)

    # TODO: Do we need get_outputs()?


ext_modules = cythonize(os.path.join('.', 'astra', '*.pyx'),
                        include_path=include_path,
                        build_dir=build_dir,
                        language_level=3)
cmdclass = {'build_ext': build_ext, 'set_dist_dir': SetDistDirCommand, 'add_extra_lib': AddExtraLibCommand }

for m in ext_modules:
    if m.name in ('astra.plugin_c', 'astra.algorithm_c'):
        m.sources.append(os.path.join('.', 'astra', 'src',
                                      'PythonPluginAlgorithm.cpp'))
    if m.name in ('astra.plugin_c'):
        m.sources.append(os.path.join('.', 'astra', 'src',
                                      'PythonPluginAlgorithmFactory.cpp'))
    if m.name in ('astra.utils'):
        m.sources.append(os.path.join('.', 'astra', 'src',
                                      'dlpack.cpp'))

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


setup(
    script_args=script_args,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
    cmdclass=cmdclass,
    packages=['astra', 'astra.plugins'],
    install_requires=['numpy', 'scipy'] + extra_install_requires,
    name='astra-toolbox',
    version='2.3.1',
    description='High-performance GPU primitives and algorithms for 2D and 3D tomography',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    project_urls={
        'Home page': 'https://astra-toolbox.com',
        'Source': 'https://github.com/astra-toolbox/astra-toolbox'
    }
)
