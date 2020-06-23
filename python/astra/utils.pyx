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
#
# distutils: language = c++
# distutils: libraries = astra

import sys
cimport numpy as np
import numpy as np
import six
if six.PY3:
    import builtins
else:
    import __builtin__ as builtins
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from cython.operator cimport dereference as deref, preincrement as inc
from cpython.version cimport PY_MAJOR_VERSION

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument
from .PyXMLDocument cimport XMLNode
from .PyIncludes cimport *

from .pythonutils import GPULink, checkArrayForLink

cdef extern from "CFloat32CustomPython.h":
    cdef cppclass CFloat32CustomPython:
        CFloat32CustomPython(arrIn)

cdef extern from "Python.h":
    void* PyLong_AsVoidPtr(object)


include "config.pxi"


cdef Config * dictToConfig(string rootname, dc) except NULL:
    cdef Config * cfg = new Config()
    cfg.initialize(rootname)
    try:
        readDict(cfg.self, dc)
    except Exception:
        del cfg
        exc = sys.exc_info()
        raise exc[0], exc[1], exc[2]
    return cfg

def convert_item(item):
    if isinstance(item, six.string_types):
        return item.encode('ascii')

    if type(item) is not dict:
        return item

    out_dict = {}
    for k in item:
        out_dict[convert_item(k)] = convert_item(item[k])
    return out_dict


def wrap_to_bytes(value):
    if isinstance(value, six.binary_type):
        return value
    s = str(value)
    if PY_MAJOR_VERSION == 3:
        s = s.encode('ascii')
    return s


def wrap_from_bytes(value):
    s = value
    if PY_MAJOR_VERSION == 3:
        s = s.decode('ascii')
    return s


cdef bool readDict(XMLNode root, _dc) except False:
    cdef XMLNode listbase
    cdef XMLNode itm
    cdef int i
    cdef int j
    cdef double* data

    dc = convert_item(_dc)
    for item in dc:
        val = dc[item]
        if isinstance(val, builtins.list) or isinstance(val, tuple):
            val = np.array(val,dtype=np.float64)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = root.addChildNode(item)
            contig_data = np.ascontiguousarray(val,dtype=np.float64)
            data = <double*>np.PyArray_DATA(contig_data)
            if val.ndim == 2:
                listbase.setContent(data, val.shape[1], val.shape[0], False)
            elif val.ndim == 1:
                listbase.setContent(data, val.shape[0])
            else:
                raise Exception("Only 1 or 2 dimensions are allowed")
        elif isinstance(val, dict):
            if item == six.b('option') or item == six.b('options') or item == six.b('Option') or item == six.b('Options'):
                readOptions(root, val)
            else:
                itm = root.addChildNode(item)
                readDict(itm, val)
        else:
            if item == six.b('type'):
                root.addAttribute(< string > six.b('type'), <string> wrap_to_bytes(val))
            else:
                if isinstance(val, builtins.bool):
                    val = int(val)
                itm = root.addChildNode(item, wrap_to_bytes(val))
    return True

cdef bool readOptions(XMLNode node, dc) except False:
    cdef XMLNode listbase
    cdef XMLNode itm
    cdef int i
    cdef int j
    cdef double* data
    for item in dc:
        val = dc[item]
        if node.hasOption(item):
            raise Exception('Duplicate Option: %s' % item)
        if isinstance(val, builtins.list) or isinstance(val, tuple):
            val = np.array(val,dtype=np.float64)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = node.addChildNode(six.b('Option'))
            listbase.addAttribute(< string > six.b('key'), < string > item)
            contig_data = np.ascontiguousarray(val,dtype=np.float64)
            data = <double*>np.PyArray_DATA(contig_data)
            if val.ndim == 2:
                listbase.setContent(data, val.shape[1], val.shape[0], False)
            elif val.ndim == 1:
                listbase.setContent(data, val.shape[0])
            else:
                raise Exception("Only 1 or 2 dimensions are allowed")
        else:
            if isinstance(val, builtins.bool):
                val = int(val)
            node.addOption(item, wrap_to_bytes(val))
    return True

cdef configToDict(Config *cfg):
    return XMLNode2dict(cfg.self)

def castString3(input):
    return input.decode('utf-8')

def castString2(input):
    return input

if six.PY3:
    castString = castString3
else:
    castString = castString2

def stringToPythonValue(inputIn):
    input = castString(inputIn)
    # matrix
    if ';' in input:
        input = input.rstrip(';')
        row_strings = input.split(';')
        col_strings = row_strings[0].split(',')
        nRows = len(row_strings)
        nCols = len(col_strings)

        out = np.empty((nRows,nCols))
        for ridx, row in enumerate(row_strings):
            col_strings = row.split(',')
            for cidx, col in enumerate(col_strings):
                out[ridx,cidx] = float(col)
        return out

    # vector
    if ',' in input:
        input = input.rstrip(',')
        items = input.split(',')
        out = np.empty(len(items))
        for idx,item in enumerate(items):
            out[idx] = float(item)
        return out

    try:
        # integer
        return int(input)
    except ValueError:
        try:
            #float
            return float(input)
        except ValueError:
            # string
            return str(input)


cdef XMLNode2dict(XMLNode node):
    cdef XMLNode subnode
    cdef list[XMLNode] nodes
    cdef list[XMLNode].iterator it
    dct = {}
    opts = {}
    if node.hasAttribute(six.b('type')):
        dct['type'] = castString(node.getAttribute(six.b('type')))
    nodes = node.getNodes()
    it = nodes.begin()
    while it != nodes.end():
        subnode = deref(it)
        if castString(subnode.getName())=="Option":
            if subnode.hasAttribute('value'):
                opts[castString(subnode.getAttribute('key'))] = stringToPythonValue(subnode.getAttribute('value'))
            else:
                opts[castString(subnode.getAttribute('key'))] = stringToPythonValue(subnode.getContent())
        else:
            dct[castString(subnode.getName())] = stringToPythonValue(subnode.getContent())
        inc(it)
    if len(opts)>0: dct['options'] = opts
    return dct

cdef CFloat32VolumeData3D* linkVolFromGeometry(CVolumeGeometry3D *pGeometry, data) except NULL:
    cdef CFloat32VolumeData3D * pDataObject3D = NULL
    geom_shape = (pGeometry.getGridSliceCount(), pGeometry.getGridRowCount(), pGeometry.getGridColCount())
    if isinstance(data, np.ndarray):
        data_shape = data.shape
    elif isinstance(data, GPULink):
        data_shape = (data.z, data.y, data.x)
    if geom_shape != data_shape:
        raise ValueError(
            "The dimensions of the data do not match those specified in the geometry: {} != {}".format(data_shape, geom_shape))

    if isinstance(data, np.ndarray):
        checkArrayForLink(data)
        pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
        pDataObject3D = new CFloat32VolumeData3DMemory(pGeometry, pCustom)
    elif isinstance(data, GPULink):
        IF HAVE_CUDA==True:
            hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
            pDataObject3D = new CFloat32VolumeData3DGPU(pGeometry, hnd)
        ELSE:
            raise NotImplementedError("CUDA support is not enabled in ASTRA")
    else:
        raise TypeError("data should be a numpy.ndarray or a GPULink object")
    return pDataObject3D

cdef CFloat32ProjectionData3D* linkProjFromGeometry(CProjectionGeometry3D *pGeometry, data) except NULL:
    cdef CFloat32ProjectionData3D * pDataObject3D = NULL
    geom_shape = (pGeometry.getDetectorRowCount(), pGeometry.getProjectionCount(), pGeometry.getDetectorColCount())
    if isinstance(data, np.ndarray):
        data_shape = data.shape
    elif isinstance(data, GPULink):
        data_shape = (data.z, data.y, data.x)
    if geom_shape != data_shape:
        raise ValueError(
            "The dimensions of the data do not match those specified in the geometry: {} != {}".format(data_shape, geom_shape))

    if isinstance(data, np.ndarray):
        checkArrayForLink(data)
        pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
        pDataObject3D = new CFloat32ProjectionData3DMemory(pGeometry, pCustom)
    elif isinstance(data, GPULink):
        IF HAVE_CUDA==True:
            hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
            pDataObject3D = new CFloat32ProjectionData3DGPU(pGeometry, hnd)
        ELSE:
            raise NotImplementedError("CUDA support is not enabled in ASTRA")
    else:
        raise TypeError("data should be a numpy.ndarray or a GPULink object")
    return pDataObject3D


