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

from __future__ import print_function

cimport cython

from . cimport PyData3DManager
from .PyData3DManager cimport CData3DManager

from .PyIncludes cimport *
import numpy as np

cimport numpy as np
np.import_array()

from . cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

from . cimport utils
from .utils import wrap_from_bytes
from .utils cimport linkVolFromGeometry, linkProjFromGeometry, createProjectionGeometry3D, createVolumeGeometry3D
from .log import AstraError

from .pythonutils import geom_size, GPULink

import operator

include "config.pxi"


cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

cdef extern from *:
    CFloat32Data3DMemory * dynamic_cast_mem "dynamic_cast<astra::CFloat32Data3DMemory*>" (CFloat32Data3D * )

cdef CFloat32Data3DMemory * dynamic_cast_mem_safe(CFloat32Data3D *obj) except NULL:
    cdef CFloat32Data3DMemory *ret = dynamic_cast_mem(obj)
    if not ret:
        raise AstraError("Not a memory 3D data object")
    return ret


def create(datatype,geometry,data=None, link=False):
    cdef CVolumeGeometry3D * pGeometry
    cdef CProjectionGeometry3D * ppGeometry
    cdef CFloat32Data3D * pDataObject3D
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
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(data_shape, geom_shape))

    if datatype == '-vol':
        pGeometry = createVolumeGeometry3D(geometry)
        if link:
            pDataObject3D = linkVolFromGeometry(pGeometry, data)
        else:
            pDataObject3D = new CFloat32VolumeData3DMemory(pGeometry)
        del pGeometry
    elif datatype == '-sino' or datatype == '-proj3d' or datatype == '-sinocone':
        ppGeometry = createProjectionGeometry3D(geometry)
        if link:
            pDataObject3D = linkProjFromGeometry(ppGeometry, data)
        else:
            pDataObject3D = new CFloat32ProjectionData3DMemory(ppGeometry)
        del ppGeometry
    else:
        raise ValueError("Invalid datatype. Please specify '-vol' or '-proj3d'")

    if not pDataObject3D.isInitialized():
        del pDataObject3D
        raise AstraError("Couldn't initialize data object", append_log=True)

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
        raise AstraError("Not a known data object")
    return geom

def change_geometry(i, geom):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem_safe(getObject(i))
    cdef CFloat32ProjectionData3DMemory * pDataObject2
    cdef CFloat32VolumeData3DMemory * pDataObject3
    if pDataObject.getType() == THREEPROJECTION:
        pDataObject2 = <CFloat32ProjectionData3DMemory * >pDataObject
        ppGeometry = createProjectionGeometry3D(geom)
        geom_shape = (ppGeometry.getDetectorRowCount(), ppGeometry.getProjectionCount(), ppGeometry.getDetectorColCount())
        obj_shape = (pDataObject2.getDetectorRowCount(), pDataObject2.getAngleCount(), pDataObject2.getDetectorColCount())
        if geom_shape != obj_shape:
            del ppGeometry
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(obj_shape, geom_shape))
        pDataObject2.changeGeometry(ppGeometry)
        del ppGeometry

    elif pDataObject.getType() == THREEVOLUME:
        pDataObject3 = <CFloat32VolumeData3DMemory * >pDataObject
        pGeometry = createVolumeGeometry3D(geom)
        geom_shape = (pGeometry.getGridSliceCount(), pGeometry.getGridRowCount(), pGeometry.getGridColCount())
        obj_shape = (pDataObject3.getSliceCount(), pDataObject3.getRowCount(), pDataObject3.getColCount())
        if geom_shape != obj_shape:
            del pGeometry
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(obj_shape, geom_shape))
        pDataObject3.changeGeometry(pGeometry)
        del pGeometry

    else:
        raise AstraError("Not a known data object")


cdef fillDataObject(CFloat32Data3DMemory * obj, data):
    if data is None:
        fillDataObjectScalar(obj, 0)
    else:
        if isinstance(data, np.ndarray):
            obj_shape = (obj.getDepth(), obj.getHeight(), obj.getWidth())
            if data.shape != obj_shape:
                raise ValueError("The dimensions of the data {} do not match those "
                                 "specified in the geometry {}".format(data.shape, obj_shape))
            fillDataObjectArray(obj, np.ascontiguousarray(data,dtype=np.float32))
        else:
            fillDataObjectScalar(obj, np.float32(data))

cdef fillDataObjectScalar(CFloat32Data3DMemory * obj, float s):
    cdef size_t i
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
        raise AstraError("Data object not found")
    if not pDataObject.isInitialized():
        raise AstraError("Data object not initialized properly")
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
    print(wrap_from_bytes(man3d.info()))
