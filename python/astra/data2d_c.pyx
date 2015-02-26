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
from cython cimport view

cimport PyData2DManager
from .PyData2DManager cimport CData2DManager

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

import numpy as np

cimport numpy as np
np.import_array()


from .PyIncludes cimport *
cimport utils
from .utils import wrap_from_bytes


cdef CData2DManager * man2d = <CData2DManager * >PyData2DManager.getSingletonPtr()


def clear():
    man2d.clear()


def delete(ids):
    try:
        for i in ids:
            man2d.remove(i)
    except TypeError:
        man2d.remove(ids)


def create(datatype, geometry, data=None):
    cdef XMLDocument * xml
    cdef Config cfg
    cdef CVolumeGeometry2D * pGeometry
    cdef CProjectionGeometry2D * ppGeometry
    cdef CFloat32Data2D * pDataObject2D
    if datatype == '-vol':
        xml = utils.dict2XML(six.b('VolumeGeometry'), geometry)
        cfg.self = xml.getRootNode()
        pGeometry = new CVolumeGeometry2D()
        if not pGeometry.initialize(cfg):
            del xml
            del pGeometry
            raise Exception('Geometry class not initialized.')
        pDataObject2D = <CFloat32Data2D * > new CFloat32VolumeData2D(pGeometry)
        del xml
        del pGeometry
    elif datatype == '-sino':
        xml = utils.dict2XML(six.b('ProjectionGeometry'), geometry)
        cfg.self = xml.getRootNode()
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == 'sparse_matrix'):
            ppGeometry = <CProjectionGeometry2D * >new CSparseMatrixProjectionGeometry2D()
        elif (tpe == 'fanflat'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatProjectionGeometry2D()
        elif (tpe == 'fanflat_vec'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatVecProjectionGeometry2D()
        else:
            ppGeometry = <CProjectionGeometry2D * >new CParallelProjectionGeometry2D()
        if not ppGeometry.initialize(cfg):
            del xml
            del ppGeometry
            raise Exception('Geometry class not initialized.')
        pDataObject2D = <CFloat32Data2D * > new CFloat32ProjectionData2D(ppGeometry)
        del ppGeometry
        del xml
    else:
        raise Exception("Invalid datatype.  Please specify '-vol' or '-sino'.")

    if not pDataObject2D.isInitialized():
        del pDataObject2D
        raise Exception("Couldn't initialize data object.")

    fillDataObject(pDataObject2D, data)

    return man2d.store(pDataObject2D)

cdef fillDataObject(CFloat32Data2D * obj, data):
    if data is None:
        fillDataObjectScalar(obj, 0)
    else:
        if isinstance(data, np.ndarray):
            fillDataObjectArray(obj, np.ascontiguousarray(data,dtype=np.float32))
        else:
            fillDataObjectScalar(obj, np.float32(data))

cdef fillDataObjectScalar(CFloat32Data2D * obj, float s):
    cdef int i
    for i in range(obj.getSize()):
        obj.getData()[i] = s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef fillDataObjectArray(CFloat32Data2D * obj, float [:,::1] data):
    if (not data.shape[0] == obj.getHeight()) or (not data.shape[1] == obj.getWidth()):
        raise Exception(
            "The dimensions of the data do not match those specified in the geometry.")
    cdef float [:,::1] cView =  <float[:data.shape[0],:data.shape[1]]> obj.getData2D()[0]
    cView[:] = data

cdef CFloat32Data2D * getObject(i) except NULL:
    cdef CFloat32Data2D * pDataObject = man2d.get(i)
    if pDataObject == NULL:
        raise Exception("Data object not found")
    if not pDataObject.isInitialized():
        raise Exception("Data object not initialized properly.")
    return pDataObject


def store(i, data):
    cdef CFloat32Data2D * pDataObject = getObject(i)
    fillDataObject(pDataObject, data)


def get_geometry(i):
    cdef CFloat32Data2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == PROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        geom = utils.createProjectionGeometryStruct(pDataObject2.getGeometry())
    elif pDataObject.getType() == VOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        geom = utils.createVolumeGeometryStruct(pDataObject3.getGeometry())
    else:
        raise Exception("Not a known data object")
    return geom


def change_geometry(i, geom):
    cdef XMLDocument * xml
    cdef Config cfg
    cdef CVolumeGeometry2D * pGeometry
    cdef CProjectionGeometry2D * ppGeometry
    cdef CFloat32Data2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == PROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        xml = utils.dict2XML(six.b('ProjectionGeometry'), geom)
        cfg.self = xml.getRootNode()
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == 'sparse_matrix'):
            ppGeometry = <CProjectionGeometry2D * >new CSparseMatrixProjectionGeometry2D()
        elif (tpe == 'fanflat'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatProjectionGeometry2D()
        elif (tpe == 'fanflat_vec'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatVecProjectionGeometry2D()
        else:
            ppGeometry = <CProjectionGeometry2D * >new CParallelProjectionGeometry2D()
        if not ppGeometry.initialize(cfg):
            del xml
            del ppGeometry
            raise Exception('Geometry class not initialized.')
        if (ppGeometry.getDetectorCount() != pDataObject2.getDetectorCount() or ppGeometry.getProjectionAngleCount() != pDataObject2.getAngleCount()):
            del ppGeometry
            del xml
            raise Exception(
                "The dimensions of the data do not match those specified in the geometry.")
        pDataObject2.changeGeometry(ppGeometry)
        del ppGeometry
        del xml
    elif pDataObject.getType() == VOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        xml = utils.dict2XML(six.b('VolumeGeometry'), geom)
        cfg.self = xml.getRootNode()
        pGeometry = new CVolumeGeometry2D()
        if not pGeometry.initialize(cfg):
            del xml
            del pGeometry
            raise Exception('Geometry class not initialized.')
        if (pGeometry.getGridColCount() != pDataObject3.getWidth() or pGeometry.getGridRowCount() != pDataObject3.getHeight()):
            del xml
            del pGeometry
            raise Exception(
                'The dimensions of the data do not match those specified in the geometry.')
        pDataObject3.changeGeometry(pGeometry)
        del xml
        del pGeometry
    else:
        raise Exception("Not a known data object")

@cython.boundscheck(False)
@cython.wraparound(False)
def get(i):
    cdef CFloat32Data2D * pDataObject = getObject(i)
    outArr = np.empty((pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef float [:,::1] mView = outArr
    cdef float [:,::1] cView =  <float[:outArr.shape[0],:outArr.shape[1]]> pDataObject.getData2D()[0]
    mView[:] = cView
    return outArr

def get_shared(i):
    cdef CFloat32Data2D * pDataObject = getObject(i)
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> pDataObject.getHeight()
    shape[1] = <np.npy_intp> pDataObject.getWidth()
    return np.PyArray_SimpleNewFromData(2,shape,np.NPY_FLOAT32,<void *>pDataObject.getData2D()[0])


def get_single(i):
    raise Exception("Not yet implemented")


def info():
    six.print_(wrap_from_bytes(man2d.info()))
