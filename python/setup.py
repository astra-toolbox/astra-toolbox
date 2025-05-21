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

from setuptools import Command, Extension, setup
from setuptools.command.build import build

from Cython.Distutils import build_ext

import argparse
import sys
import glob


# Load configuration data from pyproject.toml's tool.astra section,
# based on the value of the ASTRA_CONFIG environment variable.

extra_install_requires = []
extra_lib = []
use_cuda = False

# We require to be called in specific ways, so that we can find the ASTRA library and sources.
# Our Makefile and installation scripts take care of this. Detect if we are called in another way.
tool_profile = os.environ.get('ASTRA_CONFIG')
if not tool_profile:
    raise RuntimeError("The ASTRA python package is not intended to be installed this way. See the ASTRA installation instructions.")


try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('pyproject.toml', mode="rb") as F:
    config = tomllib.load(F)
try:
    section = config['tool']['astra'][tool_profile]
except KeyError:
    raise KeyError(f"Configuration [tool.astra.{tool_profile}] not found in pyproject.toml")

if 'install_requires' in section:
    deps = section['install_requires']
    if isinstance(deps, str):
        deps = [ deps ]
    extra_install_requires += deps
if 'extra_lib' in section:
    extra_lib = section['extra_lib']
    if isinstance(extra_lib, str):
        extra_lib = [ extra_lib ]
if 'cuda' in section:
    use_cuda = section['cuda']

# Write HAVE_CUDA to config.pxi:

cfg_string = 'DEF HAVE_CUDA=' + str(use_cuda) + '\n'
update_cfg = True

self_path = os.path.dirname(os.path.abspath(__file__))

cfg_file = os.path.join(self_path, 'astra', 'config.pxi')

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



# Custom setuptools (sub)command to install extra file into wheel

cmdclass = {}

class AddExtraLibCommand(Command):
    user_options = [
    ]
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # TODO: We could also just copy all so/dll files from the astra/
        # directory to the right place?
        # TODO: Or re-evaluate simply adding these to package_data
        if extra_lib:
            import glob
            import shutil
            build_ext = self.get_finalized_command('build_ext')
            dst = os.path.join(build_ext.build_lib, 'astra')
            for F in extra_lib:
                for src in glob.glob(F):
                    print("Installing", src, "to", dst)
                    shutil.copy2(src, dst)

# Prepare Cython modules

def prepare_ext_modules():
    pyxfiles = glob.glob(os.path.join('.', 'astra', '*.pyx'))
    ext_modules = []
    for filename in pyxfiles:
        modulename = 'astra.' + os.path.splitext(os.path.basename(filename))[0]
        sources = [filename]
        if modulename in ('astra.plugin_c', 'astra.algorithm_c'):
            sources.append(os.path.join('astra', 'src',
                                  'PythonPluginAlgorithm.cpp'))
        if modulename in ('astra.plugin_c'):
            sources.append(os.path.join('astra', 'src',
                                  'PythonPluginAlgorithmFactory.cpp'))
        if modulename in ('astra.utils'):
            sources.append(os.path.join('astra', 'src',
                                  'dlpack.cpp'))
        ext = Extension(modulename, sources=sources, libraries=["astra"])
        assert not hasattr(ext, 'cython_directives')
        ext.cython_directives = {'language_level': "3"}
        ext_modules.append(ext)

    return ext_modules

# See https://github.com/pypa/setuptools/discussions/3762
class CustomBuild(build):
    sub_commands = build.sub_commands + [('add_extra_lib', None)]

cmdclass = {'build': CustomBuild, 'build_ext': build_ext, 'add_extra_lib': AddExtraLibCommand }

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    ext_modules=prepare_ext_modules(),
    include_dirs=[np.get_include()],
    cmdclass=cmdclass,
    packages=['astra', 'astra.plugins'],
    install_requires=['numpy', 'scipy'] + extra_install_requires,
    name='astra-toolbox',
    version='2.3.0',
    description='High-performance GPU primitives and algorithms for 2D and 3D tomography',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    project_urls={
        'Home page': 'https://astra-toolbox.com',
        'Source': 'https://github.com/astra-toolbox/astra-toolbox'
    }
)
