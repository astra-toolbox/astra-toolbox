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

include "config.pxi"
import six
from .utils import wrap_from_bytes

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport PyIndexManager
from .PyIndexManager cimport CAstraObjectManagerBase

cdef extern from "astra/Globals.h" namespace "astra":
    int getVersion()
    string getVersionString()
    bool cudaEnabled()

IF HAVE_CUDA==True:
  cdef extern from "../cuda/2d/darthelper.h" namespace "astraCUDA":
      bool setGPUIndex(int)
ELSE:
  def setGPUIndex():
    pass
cdef extern from "astra/CompositeGeometryManager.h" namespace "astra":
    cdef cppclass SGPUParams:
        vector[int] GPUIndices
        size_t memory
cdef extern from "astra/CompositeGeometryManager.h" namespace "astra::CCompositeGeometryManager":
    void setGlobalGPUParams(SGPUParams&)


def credits():
    six.print_("""The ASTRA Toolbox has been developed at the University of Antwerp and CWI, Amsterdam by
 * Prof. dr. Joost Batenburg
 * Prof. dr. Jan Sijbers
 * Dr. Jeroen Bedorf
 * Dr. Folkert Bleichrodt
 * Dr. Andrei Dabravolski
 * Dr. Willem Jan Palenstijn
 * Dr. Tom Roelandts
 * Dr. Wim van Aarle
 * Dr. Gert Van Gompel
 * Sander van der Maar, MSc.
 * Gert Merckx, MSc.
 * Daan Pelt, MSc.""")


def use_cuda():
    return cudaEnabled()


def version(printToScreen=False):
    if printToScreen:
        six.print_(wrap_from_bytes(getVersionString()))
    else:
        return getVersion()

IF HAVE_CUDA==True:
  def set_gpu_index(idx, memory=0):
    import collections
    cdef SGPUParams params
    if use_cuda()==True:
        if not isinstance(idx, collections.Iterable) or isinstance(idx, six.string_types + (six.text_type,six.binary_type)):
            idx = (idx,)
        params.memory = memory
        params.GPUIndices = idx
        setGlobalGPUParams(params)
        ret = setGPUIndex(params.GPUIndices[0])
        if not ret:
            six.print_("Failed to set GPU " + str(params.GPUIndices[0]))
ELSE:
  def set_gpu_index(idx, memory=0):
    raise NotImplementedError("CUDA support is not enabled in ASTRA")

def delete(ids):
    import collections
    cdef CAstraObjectManagerBase* ptr
    if not isinstance(ids, collections.Iterable) or isinstance(ids, six.string_types + (six.text_type,six.binary_type)):
        ids = (ids,)
    for i in ids:
        ptr = PyIndexManager.getSingletonPtr().get(i)
        if ptr:
            ptr.remove(i)

def info(ids):
    import collections
    cdef CAstraObjectManagerBase* ptr
    if not isinstance(ids, collections.Iterable) or isinstance(ids, six.string_types + (six.text_type,six.binary_type)):
        ids = (ids,)
    for i in ids:
        ptr = PyIndexManager.getSingletonPtr().get(i)
        if ptr:
            s = ptr.getType() + six.b("\t") + ptr.getInfo(i)
            six.print_(wrap_from_bytes(s))
