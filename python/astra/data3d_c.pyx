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

import six

cimport cython

cimport PyData3DManager
from .PyData3DManager cimport CData3DManager

from .PyIncludes cimport *
import numpy as np

cimport numpy as np
np.import_array()

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

cimport utils
from .utils import wrap_from_bytes

from .pythonutils import geom_size, GPULink

import operator

from six.moves import reduce

include "config.pxi"

cdef extern from "Python.h":
    void* PyLong_AsVoidPtr(object)


cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

cdef extern from *:
    CFloat32Data3DMemory * dynamic_cast_mem "dynamic_cast<astra::CFloat32Data3DMemory*>" (CFloat32Data3D * )

cdef CFloat32Data3DMemory * dynamic_cast_mem_safe(CFloat32Data3D *obj) except NULL:
    cdef CFloat32Data3DMemory *ret = dynamic_cast_mem(obj)
    if not ret:
        raise RuntimeError("Not a memory 3D data object")
    return ret

cdef extern from "CFloat32CustomPython.h":
    cdef cppclass CFloat32CustomPython:
        CFloat32CustomPython(arrIn)

def create(datatype,geometry,data=None, link=False):
    cdef Config *cfg
    cdef CVolumeGeometry3D * pGeometry
    cdef CProjectionGeometry3D * ppGeometry
    cdef CFloat32Data3D * pDataObject3D
    cdef CConeProjectionGeometry3D* pppGeometry
    cdef CFloat32CustomMemory * pCustom = NULL
    IF HAVE_CUDA==True:
        cdef MemHandle3D hnd

    if link:
        geom_shape = geom_size(geometry)
        if isinstance(data, np.ndarray):
            data_shape = data.shape
        elif isinstance(data, GPULink):
            data_shape = ( data.z, data.y, data.x )
        else:
            raise TypeError("data should be a numpy.ndarray or a GPULink object")
        if geom_shape != data_shape:
            raise ValueError("The dimensions of the data do not match those specified in the geometry: {} != {}".format(data_shape, geom_shape))

    if datatype == '-vol':
        cfg = utils.dictToConfig(six.b('VolumeGeometry'), geometry)
        pGeometry = new CVolumeGeometry3D()
        if not pGeometry.initialize(cfg[0]):
            del cfg
            del pGeometry
            raise RuntimeError('Geometry class not initialized.')
        if link:
            if isinstance(data, np.ndarray):
                pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
                pDataObject3D = <CFloat32Data3D * > new CFloat32VolumeData3DMemory(pGeometry, pCustom)
            elif isinstance(data, GPULink):
                IF HAVE_CUDA==True:
                    s = geom_size(geometry)
                    hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
                    pDataObject3D = <CFloat32Data3D * > new CFloat32VolumeData3DGPU(pGeometry, hnd)
                ELSE:
                    raise NotImplementedError("CUDA support is not enabled in ASTRA")
            else:
                raise TypeError("data should be a numpy.ndarray or a GPULink object")
        else:
            pDataObject3D = <CFloat32Data3D * > new CFloat32VolumeData3DMemory(pGeometry)
        del cfg
        del pGeometry
    elif datatype == '-sino' or datatype == '-proj3d' or datatype == '-sinocone':
        cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geometry)
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == "parallel3d"):
            ppGeometry = <CProjectionGeometry3D*> new CParallelProjectionGeometry3D();
        elif (tpe == "parallel3d_vec"):
            ppGeometry = <CProjectionGeometry3D*> new CParallelVecProjectionGeometry3D();
        elif (tpe == "cone"):
            ppGeometry = <CProjectionGeometry3D*> new CConeProjectionGeometry3D();
        elif (tpe == "cone_vec"):
            ppGeometry = <CProjectionGeometry3D*> new CConeVecProjectionGeometry3D();
        else:
            raise ValueError("Invalid geometry type.")

        if not ppGeometry.initialize(cfg[0]):
            del cfg
            del ppGeometry
            raise RuntimeError('Geometry class not initialized.')
        if link:
            if isinstance(data, np.ndarray):
                pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
                pDataObject3D = <CFloat32Data3D * > new CFloat32ProjectionData3DMemory(ppGeometry, pCustom)
            elif isinstance(data, GPULink):
                IF HAVE_CUDA==True:
                    s = geom_size(geometry)
                    hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
                    pDataObject3D = <CFloat32Data3D * > new CFloat32ProjectionData3DGPU(ppGeometry, hnd)
                ELSE:
                    raise NotImplementedError("CUDA support is not enabled in ASTRA")
            else:
                raise TypeError("data should be a numpy.ndarray or a GPULink object")
        else:
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(ppGeometry)
        del ppGeometry
        del cfg
    else:
        raise ValueError("Invalid datatype.  Please specify '-vol' or '-proj3d'.")

    if not pDataObject3D.isInitialized():
        del pDataObject3D
        raise RuntimeError("Couldn't initialize data object.")

    if not link:
        fillDataObject(dynamic_cast_mem_safe(pDataObject3D), data)

    return man3d.store(<CFloat32Data3D*>pDataObject3D)

def get_geometry(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem_safe(getObject(i))
    cdef CFloat32ProjectionData3DMemory * pDataObject2
    cdef CFloat32VolumeData3DMemory * pDataObject3
    if pDataObject.getType() == THREEPROJECTION:
        pDataObject2 = <CFloat32ProjectionData3DMemory * >pDataObject
        geom = utils.configToDict(pDataObject2.getGeometry().getConfiguration())
    elif pDataObject.getType() == THREEVOLUME:
        pDataObject3 = <CFloat32VolumeData3DMemory * >pDataObject
        geom = utils.configToDict(pDataObject3.getGeometry().getConfiguration())
    else:
        raise RuntimeError("Not a known data object")
    return geom

def change_geometry(i, geom):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem_safe(getObject(i))
    cdef CFloat32ProjectionData3DMemory * pDataObject2
    cdef CFloat32VolumeData3DMemory * pDataObject3
    if pDataObject.getType() == THREEPROJECTION:
        pDataObject2 = <CFloat32ProjectionData3DMemory * >pDataObject
        # TODO: Reduce code duplication here
        cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geom)
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == "parallel3d"):
            ppGeometry = <CProjectionGeometry3D*> new CParallelProjectionGeometry3D();
        elif (tpe == "parallel3d_vec"):
            ppGeometry = <CProjectionGeometry3D*> new CParallelVecProjectionGeometry3D();
        elif (tpe == "cone"):
            ppGeometry = <CProjectionGeometry3D*> new CConeProjectionGeometry3D();
        elif (tpe == "cone_vec"):
            ppGeometry = <CProjectionGeometry3D*> new CConeVecProjectionGeometry3D();
        else:
            raise ValueError("Invalid geometry type.")
        if not ppGeometry.initialize(cfg[0]):
            del cfg
            del ppGeometry
            raise RuntimeError('Geometry class not initialized.')
        del cfg
        geom_shape = (ppGeometry.getDetectorRowCount(), ppGeometry.getProjectionCount(), ppGeometry.getDetectorColCount())
        obj_shape = (pDataObject2.getDetectorRowCount(), pDataObject2.getAngleCount(), pDataObject2.getDetectorColCount())
        if geom_shape != obj_shape:
            del ppGeometry
            raise ValueError(
                "The dimensions of the data do not match those specified in the geometry: {} != {}".format(obj_shape, geom_shape))
        pDataObject2.changeGeometry(ppGeometry)
        del ppGeometry

    elif pDataObject.getType() == THREEVOLUME:
        pDataObject3 = <CFloat32VolumeData3DMemory * >pDataObject
        cfg = utils.dictToConfig(six.b('VolumeGeometry'), geom)
        pGeometry = new CVolumeGeometry3D()
        if not pGeometry.initialize(cfg[0]):
            del cfg
            del pGeometry
            raise RuntimeError('Geometry class not initialized.')
        del cfg
        geom_shape = (pGeometry.getGridSliceCount(), pGeometry.getGridRowCount(), pGeometry.getGridColCount())
        obj_shape = (pDataObject3.getSliceCount(), pDataObject3.getRowCount(), pDataObject3.getColCount())
        if geom_shape != obj_shape:
            del pGeometry
            raise ValueError(
                "The dimensions of the data do not match those specified in the geometry.".format(obj_shape, geom_shape))
        pDataObject3.changeGeometry(pGeometry)
        del pGeometry

    else:
        raise RuntimeError("Not a known data object")


cdef fillDataObject(CFloat32Data3DMemory * obj, data):
    if data is None:
        fillDataObjectScalar(obj, 0)
    else:
        if isinstance(data, np.ndarray):
            obj_shape = (obj.getDepth(), obj.getHeight(), obj.getWidth())
            if data.shape != obj_shape:
                raise ValueError(
                  "The dimensions of the data do not match those specified in the geometry: {} != {}".format(data.shape, obj_shape))
            fillDataObjectArray(obj, np.ascontiguousarray(data,dtype=np.float32))
        else:
            fillDataObjectScalar(obj, np.float32(data))

cdef fillDataObjectScalar(CFloat32Data3DMemory * obj, float s):
    cdef int i
    for i in range(obj.getSize()):
        obj.getData()[i] = s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef fillDataObjectArray(CFloat32Data3DMemory * obj, float [:,:,::1] data):
    cdef float [:,:,::1] cView = <float[:data.shape[0],:data.shape[1],:data.shape[2]]> obj.getData()
    cView[:] = data

cdef CFloat32Data3D * getObject(i) except NULL:
    cdef CFloat32Data3D * pDataObject = man3d.get(i)
    if pDataObject == NULL:
        raise ValueError("Data object not found")
    if not pDataObject.isInitialized():
        raise RuntimeError("Data object not initialized properly.")
    return pDataObject

@cython.boundscheck(False)
@cython.wraparound(False)
def get(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem_safe(getObject(i))
    outArr = np.empty((pDataObject.getDepth(),pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef float [:,:,::1] mView = outArr
    cdef float [:,:,::1] cView = <float[:outArr.shape[0],:outArr.shape[1],:outArr.shape[2]]> pDataObject.getData()
    mView[:] = cView
    return outArr

def get_shared(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem_safe(getObject(i))
    cdef np.npy_intp shape[3]
    shape[0] = <np.npy_intp> pDataObject.getDepth()
    shape[1] = <np.npy_intp> pDataObject.getHeight()
    shape[2] = <np.npy_intp> pDataObject.getWidth()
    return np.PyArray_SimpleNewFromData(3,shape,np.NPY_FLOAT32,<void *>pDataObject.getData())

def get_single(i):
    raise NotImplementedError("Not yet implemented")

def store(i,data):
    cdef CFloat32Data3D * pDataObject = getObject(i)
    fillDataObject(dynamic_cast_mem_safe(pDataObject), data)

def dimensions(i):
    cdef CFloat32Data3D * pDataObject = getObject(i)
    return (pDataObject.getDepth(),pDataObject.getHeight(),pDataObject.getWidth())

def delete(ids):
    try:
        for i in ids:
            man3d.remove(i)
    except TypeError:
        man3d.remove(ids)

def clear():
    man3d.clear()

def info():
    six.print_(wrap_from_bytes(man3d.info()))
