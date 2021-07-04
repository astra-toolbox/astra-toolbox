# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://sf.net/projects/astra-toolbox
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
#-----------------------------------------------------------------------

import sys
import os, os.path

import subprocess
import shutil
import setuptools
import glob
from distutils.version import LooseVersion
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import Cython
    if LooseVersion(Cython.__version__)<LooseVersion('0.13'): raise ImportError("Cython version should be at least 0.13")
    import numpy as np
except ImportError:
    np = None
    pass

pkgdata={}
cmdclass = { }
ext_modules = [ ]

if not 'sdist' in sys.argv and not 'egg_info' in sys.argv:
    usecuda=False
    try:
        cuda_root=os.environ['CUDA_ROOT']
        usecuda=True
    except KeyError:
        pass
    cfgToWrite = 'DEF HAVE_CUDA=' + str(usecuda) + "\n"
    cfgHasToBeUpdated = True
    try:
        cfg = open('astra/config.pxi','r')
        cfgIn = cfg.read()
        cfg.close()
        if cfgIn==cfgToWrite:
            cfgHasToBeUpdated = False
    except IOError:
        pass
    if cfgHasToBeUpdated:
        cfg = open('astra/config.pxi','w')
        cfg.write(cfgToWrite)
        cfg.close()

    # Compile ASTRA C++ library
    savedPath = os.getcwd()
    os.chdir('astra-toolbox/build/linux/')
    subprocess.call(['./autogen.sh'])
    confcall = ['./configure','--prefix={}/build/'.format(savedPath)]
    if usecuda:
        confcall.append('--with-cuda={}'.format(cuda_root))
    subprocess.call(confcall)

    makecall = ['make']
    try:
        makeopts = os.environ['MAKEOPTS'].split()
        makecall.extend(makeopts)
    except KeyError:
        pass
    makecall.append('install')
    subprocess.call(makecall)

    subprocess.call(['make','distclean'])

    os.chdir(savedPath)

    # Copy compiled libastra to module folder
    for f in glob.glob('{}/build/lib/libastra.so*'.format(savedPath)):
        shutil.copy(f,'astra/')

    # Make compiled plugin libraries link to libastra and enable them to
    # find libastra in module folder
    try:
        os.environ["LDFLAGS"] += " -Wl,-rpath '-Wl,$ORIGIN' -Lastra/ -lastra"
    except KeyError:
        os.environ["LDFLAGS"] = " -Wl,-rpath '-Wl,$ORIGIN' -Lastra/ -lastra"

    addcflags="-DASTRA_PYTHON "
    if usecuda:
        addcflags += "-DASTRA_CUDA -I{}/include".format(cuda_root)
    try:
        os.environ["CPPFLAGS"] += addcflags
    except KeyError:
        os.environ["CPPFLAGS"] = addcflags


    

    ext_modules = cythonize("astra/*.pyx", language_level=2)
    cmdclass = { 'build_ext': build_ext }

    for m in ext_modules:
        if m.name == 'astra.plugin_c':
            m.sources.append('astra/src/PythonPluginAlgorithm.cpp')

    pkgdata['astra']=['libastra.so*']

reqpkgs = ["numpy","six","scipy","cython"]
incdirs = ['astra-toolbox/include/']
if np:
    incdirs.append(np.get_include())

setup (name = 'astra-toolbox',
       version = '1.8b4',
       description = 'Python interface to the ASTRA Toolbox',
       author='D.M. Pelt',
       author_email='D.M.Pelt@cwi.nl',
       url='http://www.github.com/astra-toolbox/astra-toolbox',
       license='GPLv3',
       ext_modules = ext_modules,
       include_dirs=incdirs,
       cmdclass = cmdclass,
       packages=['astra'],
       package_data=pkgdata,
       install_requires=reqpkgs,
       requires=reqpkgs,
	)
