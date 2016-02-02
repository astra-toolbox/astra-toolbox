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
# -----------------------------------------------------------------------
#
# distutils: language = c++
# distutils: libraries = astra

import six
import inspect

from libcpp.string cimport string
from libcpp cimport bool

cdef CPythonPluginAlgorithmFactory *fact = getSingletonPtr()

from . import utils

cdef extern from "src/PythonPluginAlgorithm.h" namespace "astra":
    cdef cppclass CPythonPluginAlgorithmFactory:
        bool registerPlugin(string className)
        bool registerPlugin(string name, string className)
        bool registerPluginClass(object className)
        bool registerPluginClass(string name, object className)
        object getRegistered()
        string getHelp(string &name)

cdef extern from "src/PythonPluginAlgorithm.h" namespace "astra::CPythonPluginAlgorithmFactory":
    cdef CPythonPluginAlgorithmFactory* getSingletonPtr()

cdef extern from "astra/PluginAlgorithm.h" namespace "astra::CPluginAlgorithmFactory":
    # NB: Using wrong pointer type here for convenience
    cdef void registerFactory(CPythonPluginAlgorithmFactory *)

def register(className, name=None):
    if inspect.isclass(className):
        if name==None:
            fact.registerPluginClass(className)
        else:
            fact.registerPluginClass(six.b(name), className)
    else:
        if name==None:
            fact.registerPlugin(six.b(className))
        else:
            fact.registerPlugin(six.b(name), six.b(className))

def get_registered():
    return fact.getRegistered()

def get_help(name):
    return utils.wrap_from_bytes(fact.getHelp(six.b(name)))

# Register python plugin factory with astra
registerFactory(fact)
