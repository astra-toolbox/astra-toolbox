#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pyastratoolbox/
#
#
#This file is part of the Python interface to the
#All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
#The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
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

cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

cdef extern from *:
    CFloat32Data3DMemory * dynamic_cast_mem "dynamic_cast<astra::CFloat32Data3DMemory*>" (CFloat32Data3D * ) except NULL

def create(datatype,geometry,data=None):
    cdef XMLDocument * xml
    cdef Config cfg
    cdef CVolumeGeometry3D * pGeometry
    cdef CProjectionGeometry3D * ppGeometry
    cdef CFloat32Data3DMemory * pDataObject3D
    cdef CConeProjectionGeometry3D* pppGeometry
    if datatype == '-vol':
        xml = utils.dict2XML(six.b('VolumeGeometry'), geometry)
        cfg.self = xml.getRootNode()
        pGeometry = new CVolumeGeometry3D()
        if not pGeometry.initialize(cfg):
            del xml
            del pGeometry
            raise Exception('Geometry class not initialized.')
        pDataObject3D = <CFloat32Data3DMemory * > new CFloat32VolumeData3DMemory(pGeometry)
        del xml
        del pGeometry
    elif datatype == '-sino' or datatype == '-proj3d':
        xml = utils.dict2XML(six.b('ProjectionGeometry'), geometry)
        cfg.self = xml.getRootNode()
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
            raise Exception("Invalid geometry type.")

        if not ppGeometry.initialize(cfg):
            del xml
            del ppGeometry
            raise Exception('Geometry class not initialized.')
        pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(ppGeometry)
        del ppGeometry
        del xml
    elif datatype == "-sinocone":
        xml = utils.dict2XML(six.b('ProjectionGeometry'), geometry)
        cfg.self = xml.getRootNode()
        pppGeometry = new CConeProjectionGeometry3D()
        if not pppGeometry.initialize(cfg):
            del xml
            del pppGeometry
            raise Exception('Geometry class not initialized.')
        pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(pppGeometry)
    else:
        raise Exception("Invalid datatype.  Please specify '-vol' or '-proj3d'.")

    if not pDataObject3D.isInitialized():
        del pDataObject3D
        raise Exception("Couldn't initialize data object.")

    fillDataObject(pDataObject3D, data)

    pDataObject3D.updateStatistics()

    return man3d.store(<CFloat32Data3D*>pDataObject3D)


cdef fillDataObject(CFloat32Data3DMemory * obj, data):
    if data is None:
        fillDataObjectScalar(obj, 0)
    else:
        if isinstance(data, np.ndarray):
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
    if (not data.shape[0] == obj.getDepth()) or (not data.shape[1] == obj.getHeight()) or (not data.shape[2] == obj.getWidth()):
        raise Exception(
            "The dimensions of the data do not match those specified in the geometry.")
    cdef float [:,:,::1] cView = <float[:data.shape[0],:data.shape[1],:data.shape[2]]> obj.getData3D()[0][0]
    cView[:] = data

cdef CFloat32Data3D * getObject(i) except NULL:
    cdef CFloat32Data3D * pDataObject = man3d.get(i)
    if pDataObject == NULL:
        raise Exception("Data object not found")
    if not pDataObject.isInitialized():
        raise Exception("Data object not initialized properly.")
    return pDataObject

@cython.boundscheck(False)
@cython.wraparound(False)
def get(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    outArr = np.empty((pDataObject.getDepth(),pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef float [:,:,::1] mView = outArr
    cdef float [:,:,::1] cView = <float[:outArr.shape[0],:outArr.shape[1],:outArr.shape[2]]> pDataObject.getData3D()[0][0]
    mView[:] = cView
    return outArr

def get_shared(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    outArr = np.empty((pDataObject.getDepth(),pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef np.npy_intp shape[3]
    shape[0] = <np.npy_intp> pDataObject.getDepth()
    shape[1] = <np.npy_intp> pDataObject.getHeight()
    shape[2] = <np.npy_intp> pDataObject.getWidth()
    return np.PyArray_SimpleNewFromData(3,shape,np.NPY_FLOAT32,<void *>pDataObject.getData3D()[0][0])

def get_single(i):
    raise Exception("Not yet implemented")

def store(i,data):
    cdef CFloat32Data3D * pDataObject = getObject(i)
    fillDataObject(dynamic_cast_mem(pDataObject), data)

def dimensions(i):
    cdef CFloat32Data3D * pDataObject = getObject(i)
    return (pDataObject.getWidth(),pDataObject.getHeight(),pDataObject.getDepth())

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
