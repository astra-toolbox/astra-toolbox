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

cimport PyProjector2DManager
from .PyProjector2DManager cimport CProjector2DManager

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

import numpy as np

cimport numpy as np
np.import_array()


from .PyIncludes cimport *
cimport utils
from .utils import wrap_from_bytes

from .pythonutils import geom_size

import operator

from six.moves import reduce

cdef CData2DManager * man2d = <CData2DManager * >PyData2DManager.getSingletonPtr()
cdef CProjector2DManager * manProj = <CProjector2DManager * >PyProjector2DManager.getSingletonPtr()


cdef extern from "CFloat32CustomPython.h":
    cdef cppclass CFloat32CustomPython:
        CFloat32CustomPython(arrIn)

def clear():
    man2d.clear()


def delete(ids):
    try:
        for i in ids:
            man2d.remove(i)
    except TypeError:
        man2d.remove(ids)


def create(datatype, geometry, data=None, link=False):
    cdef Config *cfg
    cdef CVolumeGeometry2D * pGeometry
    cdef CProjectionGeometry2D * ppGeometry
    cdef CFloat32Data2D * pDataObject2D
    cdef CFloat32CustomMemory * pCustom

    if link and data.shape!=geom_size(geometry):
        raise Exception("The dimensions of the data do not match those specified in the geometry.")

    if datatype == '-vol':
        cfg = utils.dictToConfig(six.b('VolumeGeometry'), geometry)
        pGeometry = new CVolumeGeometry2D()
        if not pGeometry.initialize(cfg[0]):
            del cfg
            del pGeometry
            raise Exception('Geometry class not initialized.')
        if link:
            pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
            pDataObject2D = <CFloat32Data2D * > new CFloat32VolumeData2D(pGeometry, pCustom)
        else:
            pDataObject2D = <CFloat32Data2D * > new CFloat32VolumeData2D(pGeometry)
        del cfg
        del pGeometry
    elif datatype == '-sino':
        cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geometry)
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == 'sparse_matrix'):
            ppGeometry = <CProjectionGeometry2D * >new CSparseMatrixProjectionGeometry2D()
        elif (tpe == 'fanflat'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatProjectionGeometry2D()
        elif (tpe == 'fanflat_vec'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatVecProjectionGeometry2D()
        else:
            ppGeometry = <CProjectionGeometry2D * >new CParallelProjectionGeometry2D()
        if not ppGeometry.initialize(cfg[0]):
            del cfg
            del ppGeometry
            raise Exception('Geometry class not initialized.')
        if link:
            pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
            pDataObject2D = <CFloat32Data2D * > new CFloat32ProjectionData2D(ppGeometry, pCustom)
        else:
            pDataObject2D = <CFloat32Data2D * > new CFloat32ProjectionData2D(ppGeometry)
        del ppGeometry
        del cfg
    else:
        raise Exception("Invalid datatype.  Please specify '-vol' or '-sino'.")

    if not pDataObject2D.isInitialized():
        del pDataObject2D
        raise Exception("Couldn't initialize data object.")

    if not link: fillDataObject(pDataObject2D, data)

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
    if pDataObject.getType() == TWOPROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        geom = utils.configToDict(pDataObject2.getGeometry().getConfiguration())
    elif pDataObject.getType() == TWOVOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        geom = utils.configToDict(pDataObject3.getGeometry().getConfiguration())
    else:
        raise Exception("Not a known data object")
    return geom

cdef CProjector2D * getProjector(i) except NULL:
    cdef CProjector2D * proj = manProj.get(i)
    if proj == NULL:
        raise Exception("Projector not initialized.")
    if not proj.isInitialized():
        raise Exception("Projector not initialized.")
    return proj

def check_compatible(i, proj_id):
    cdef CProjector2D * proj = getProjector(proj_id)
    cdef CFloat32Data2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == TWOPROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        return pDataObject2.getGeometry().isEqual(proj.getProjectionGeometry())
    elif pDataObject.getType() == TWOVOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        return pDataObject3.getGeometry().isEqual(proj.getVolumeGeometry())
    else:
        raise Exception("Not a known data object")

def change_geometry(i, geom):
    cdef Config *cfg
    cdef CVolumeGeometry2D * pGeometry
    cdef CProjectionGeometry2D * ppGeometry
    cdef CFloat32Data2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == TWOPROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geom)
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == 'sparse_matrix'):
            ppGeometry = <CProjectionGeometry2D * >new CSparseMatrixProjectionGeometry2D()
        elif (tpe == 'fanflat'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatProjectionGeometry2D()
        elif (tpe == 'fanflat_vec'):
            ppGeometry = <CProjectionGeometry2D * >new CFanFlatVecProjectionGeometry2D()
        else:
            ppGeometry = <CProjectionGeometry2D * >new CParallelProjectionGeometry2D()
        if not ppGeometry.initialize(cfg[0]):
            del cfg
            del ppGeometry
            raise Exception('Geometry class not initialized.')
        if (ppGeometry.getDetectorCount() != pDataObject2.getDetectorCount() or ppGeometry.getProjectionAngleCount() != pDataObject2.getAngleCount()):
            del ppGeometry
            del cfg
            raise Exception(
                "The dimensions of the data do not match those specified in the geometry.")
        pDataObject2.changeGeometry(ppGeometry)
        del ppGeometry
        del cfg
    elif pDataObject.getType() == TWOVOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        cfg = utils.dictToConfig(six.b('VolumeGeometry'), geom)
        pGeometry = new CVolumeGeometry2D()
        if not pGeometry.initialize(cfg[0]):
            del cfg
            del pGeometry
            raise Exception('Geometry class not initialized.')
        if (pGeometry.getGridColCount() != pDataObject3.getWidth() or pGeometry.getGridRowCount() != pDataObject3.getHeight()):
            del cfg
            del pGeometry
            raise Exception(
                'The dimensions of the data do not match those specified in the geometry.')
        pDataObject3.changeGeometry(pGeometry)
        del cfg
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
