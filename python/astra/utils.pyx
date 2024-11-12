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

import sys
cimport numpy as np
import numpy as np
import builtins
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.utility cimport move
from cython.operator cimport dereference as deref, preincrement as inc
from cpython.pycapsule cimport PyCapsule_IsValid

from . cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument
from .PyXMLDocument cimport XMLNode
from .PyIncludes cimport *

from .pythonutils import GPULink, checkArrayForLink
from .log import AstraError

cdef extern from "Python.h":
    void* PyLong_AsVoidPtr(object)

cdef extern from *:
    XMLConfig* dynamic_cast_XMLConfig "dynamic_cast<astra::XMLConfig*>" (Config*)

cdef extern from "src/dlpack.h":
    CFloat32VolumeData3D* getDLTensor(obj, const CVolumeGeometry3D &pGeom, string &error)
    CFloat32ProjectionData3D* getDLTensor(obj, const CProjectionGeometry3D &pGeom, string &error)



include "config.pxi"


cdef XMLConfig * dictToConfig(string rootname, dc) except NULL:
    cdef XMLConfig * cfg = new XMLConfig(rootname)
    try:
        readDict(cfg.self, dc)
    except:
        del cfg
        raise
    return cfg

def convert_item(item):
    if isinstance(item, str):
        return item.encode('ascii')

    if type(item) is not dict:
        return item

    out_dict = {}
    for k in item:
        out_dict[convert_item(k)] = convert_item(item[k])
    return out_dict


def wrap_to_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode('ascii')


def wrap_from_bytes(value):
    return value.decode('ascii')


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
                raise AstraError("Only 1 or 2 dimensions are allowed")
        elif isinstance(val, dict):
            if item == b'option' or item == b'options' or item == b'Option' or item == b'Options':
                readOptions(root, val)
            else:
                itm = root.addChildNode(item)
                readDict(itm, val)
        else:
            if item == b'type':
                root.addAttribute(< string > b'type', <string> wrap_to_bytes(val))
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
            raise AstraError('Duplicate Option: %s' % item)
        if isinstance(val, builtins.list) or isinstance(val, tuple):
            val = np.array(val,dtype=np.float64)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = node.addChildNode(b'Option')
            listbase.addAttribute(< string > b'key', < string > item)
            contig_data = np.ascontiguousarray(val,dtype=np.float64)
            data = <double*>np.PyArray_DATA(contig_data)
            if val.ndim == 2:
                listbase.setContent(data, val.shape[1], val.shape[0], False)
            elif val.ndim == 1:
                listbase.setContent(data, val.shape[0])
            else:
                raise AstraError("Only 1 or 2 dimensions are allowed")
        else:
            if isinstance(val, builtins.bool):
                val = int(val)
            node.addOption(item, wrap_to_bytes(val))
    return True

cdef configToDict(Config *cfg):
    cdef XMLConfig* xmlcfg;
    xmlcfg = dynamic_cast_XMLConfig(cfg);
    if not xmlcfg:
        return None
    return XMLNode2dict(xmlcfg.self)

def castString(input):
    return input.decode('utf-8')

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
    if node.hasAttribute(b'type'):
        dct['type'] = castString(node.getAttribute(b'type'))
    nodes = node.getNodes()
    it = nodes.begin()
    while it != nodes.end():
        subnode = deref(it)
        if castString(subnode.getName())=="Option":
            if subnode.hasAttribute(b'value'):
                opts[castString(subnode.getAttribute(b'key'))] = stringToPythonValue(subnode.getAttribute(b'value'))
            else:
                opts[castString(subnode.getAttribute(b'key'))] = stringToPythonValue(subnode.getContent())
        else:
            dct[castString(subnode.getName())] = stringToPythonValue(subnode.getContent())
        inc(it)
    if len(opts)>0: dct['options'] = opts
    return dct

def getDLPackCapsule(data):
    # backward compatibility: check if the object is a dltensor capsule already
    if PyCapsule_IsValid(data, "dltensor"):
        return data
    if not hasattr(data, "__dlpack__"):
        return None
    capsule = None
    # TODO: investigate the stream argument to __dlpack__().
    try:
        capsule = data.__dlpack__(max_version = (1,0))
    except AttributeError:
        return None
    except TypeError:
        # unsupported max_version argument raises a TypeError
        pass
    if capsule is not None:
        return capsule

    try:
        capsule = data.__dlpack__()
    except AttributeError:
        return None
    return capsule

cdef CFloat32VolumeData3D* linkVolFromGeometry(const CVolumeGeometry3D &pGeometry, data) except NULL:
    cdef CFloat32VolumeData3D * pDataObject3D = NULL
    cdef CDataStorage * pStorage
    cdef string dlerror = b""

    # TODO: investigate the stream argument to __dlpack__().
    capsule = getDLPackCapsule(data)
    if capsule is not None:
        pDataObject3D = getDLTensor(capsule, pGeometry, dlerror)
        if not pDataObject3D:
            raise ValueError("Failed to link dlpack array: " + wrap_from_bytes(dlerror))
        return pDataObject3D

    if isinstance(data, GPULink):
        geom_shape = (pGeometry.getGridSliceCount(), pGeometry.getGridRowCount(), pGeometry.getGridColCount())
        data_shape = (data.z, data.y, data.x)
        if geom_shape != data_shape:
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(data_shape, geom_shape))

        IF HAVE_CUDA==True:
            hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
            pStorage = new CDataGPU(hnd)
        ELSE:
            raise AstraError("CUDA support is not enabled in ASTRA")
        pDataObject3D = new CFloat32VolumeData3D(pGeometry, pStorage)
        return pDataObject3D

    raise TypeError("Data should be an array with DLPack support, or a GPULink object")


cdef CFloat32ProjectionData3D* linkProjFromGeometry(const CProjectionGeometry3D &pGeometry, data) except NULL:
    cdef CFloat32ProjectionData3D * pDataObject3D = NULL
    cdef CDataStorage * pStorage
    cdef string dlerror = b""

    # TODO: investigate the stream argument to __dlpack__().
    capsule = getDLPackCapsule(data)
    if capsule is not None:
        pDataObject3D = getDLTensor(capsule, pGeometry, dlerror)
        if not pDataObject3D:
            raise ValueError("Failed to link dlpack array: " + wrap_from_bytes(dlerror))
        return pDataObject3D

    if isinstance(data, GPULink):
        geom_shape = (pGeometry.getDetectorRowCount(), pGeometry.getProjectionCount(), pGeometry.getDetectorColCount())
        data_shape = (data.z, data.y, data.x)
        if geom_shape != data_shape:
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(data_shape, geom_shape))

        IF HAVE_CUDA==True:
            hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
            pStorage = new CDataGPU(hnd)
        ELSE:
            raise AstraError("CUDA support is not enabled in ASTRA")
        pDataObject3D = new CFloat32ProjectionData3D(pGeometry, pStorage)
        return pDataObject3D

    raise TypeError("Data should be an array with DLPack support, or a GPULink object")

cdef unique_ptr[CProjectionGeometry3D] createProjectionGeometry3D(geometry) except *:
    cdef XMLConfig *cfg
    cdef unique_ptr[CProjectionGeometry3D] pGeometry

    cfg = dictToConfig(b'ProjectionGeometry', geometry)
    tpe = cfg.self.getAttribute(b'type')
    pGeometry = constructProjectionGeometry3D(tpe)
    if not pGeometry:
        raise ValueError("'{}' is not a valid 3D geometry type".format(tpe))

    if not pGeometry.get().initialize(cfg[0]):
        del cfg
        raise AstraError('Geometry class could not be initialized', append_log=True)

    del cfg

    return move(pGeometry)

cdef unique_ptr[CVolumeGeometry3D] createVolumeGeometry3D(geometry) except *:
    cdef XMLConfig *cfg
    cdef CVolumeGeometry3D * pGeometry
    cfg = dictToConfig(b'VolumeGeometry', geometry)
    pGeometry = new CVolumeGeometry3D()
    if not pGeometry.initialize(cfg[0]):
        del cfg
        del pGeometry
        raise AstraError('Geometry class could not be initialized', append_log=True)

    del cfg

    return unique_ptr[CVolumeGeometry3D](pGeometry)
