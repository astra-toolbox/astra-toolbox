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
#
# distutils: language = c++
# distutils: libraries = astra

import inspect

from libcpp.string cimport string
from libcpp cimport bool

cdef CPythonPluginAlgorithmFactory *fact = getSingletonPtr()

from .utils import wrap_from_bytes, wrap_to_bytes

cdef extern from "src/PythonPluginAlgorithm.h" namespace "astra":
    cdef cppclass CPythonPluginAlgorithmFactory:
        bool registerPlugin(string className)
        bool registerPlugin(string name, string className)
        bool registerPluginClass(object className)
        bool registerPluginClass(string name, object className)
        object getRegistered()
        string getHelp(string &name)

cdef extern from "src/PythonPluginAlgorithmFactory.h" namespace "astra::CPythonPluginAlgorithmFactory":
    cdef CPythonPluginAlgorithmFactory* getSingletonPtr()

cdef extern from "astra/PluginAlgorithmFactory.h" namespace "astra::CPluginAlgorithmFactory":
    # NB: Using wrong pointer type here for convenience
    cdef void registerFactory(CPythonPluginAlgorithmFactory *)

def register(className, name=None):
    if inspect.isclass(className):
        if name==None:
            fact.registerPluginClass(className)
        else:
            fact.registerPluginClass(wrap_to_bytes(name), className)
    else:
        if name==None:
            fact.registerPlugin(wrap_to_bytes(className))
        else:
            fact.registerPlugin(wrap_to_bytes(name), wrap_to_bytes(className))

def get_registered():
    return fact.getRegistered()

def get_help(name):
    return wrap_from_bytes(fact.getHelp(wrap_to_bytes(name)))

# Register python plugin factory with astra
registerFactory(fact)
