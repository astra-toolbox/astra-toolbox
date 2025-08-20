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

from __future__ import print_function

cimport cython
from cython cimport view

from libcpp.utility cimport move

from . cimport PyData2DManager
from .PyData2DManager cimport CData2DManager

from . cimport PyProjector2DManager
from .PyProjector2DManager cimport CProjector2DManager

from . cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

import numpy as np

cimport numpy as np
np.import_array()


from .PyIncludes cimport *
from . cimport utils
from .utils import wrap_from_bytes
from .utils cimport linkVolFromGeometry2D, linkProjFromGeometry2D, createProjectionGeometry2D, createVolumeGeometry2D
from .log import AstraError

from .pythonutils import geom_size

import operator

cdef CData2DManager * man2d = <CData2DManager * >PyData2DManager.getSingletonPtr()
cdef CProjector2DManager * manProj = <CProjector2DManager * >PyProjector2DManager.getSingletonPtr()


cdef extern from "astra/SheppLogan.h" namespace "astra":
    cdef void generateSheppLogan(CFloat32VolumeData2D*, bool)

def clear():
    man2d.clear()


def delete(ids):
    try:
        for i in ids:
            man2d.remove(i)
    except TypeError:
        man2d.remove(ids)


def create(datatype, geometry, data=None, link=False):
    cdef unique_ptr[CVolumeGeometry2D] pGeometry
    cdef unique_ptr[CProjectionGeometry2D] ppGeometry
    cdef CData2D * pDataObject2D

    if link:
        geom_shape = geom_size(geometry)
        if data.shape != geom_shape:
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(data.shape, geom_shape))

    if datatype == '-vol':
        pGeometry = createVolumeGeometry2D(geometry)
        if link:
            pDataObject2D = linkVolFromGeometry2D(cython.operator.dereference(pGeometry), data)
        else:
            pDataObject2D = createCFloat32VolumeData2DMemory(move(pGeometry))
    elif datatype == '-sino':
        ppGeometry = createProjectionGeometry2D(geometry)
        if link:
            pDataObject2D = linkProjFromGeometry2D(cython.operator.dereference(ppGeometry), data)
        else:
            pDataObject2D = createCFloat32ProjectionData2DMemory(move(ppGeometry))
    else:
        raise ValueError("Invalid datatype. Please specify '-vol' or '-sino'")

    if not pDataObject2D.isInitialized():
        del pDataObject2D
        raise AstraError("Couldn't initialize data object", append_log=True)

    if not link:
        fillDataObject(pDataObject2D, data)

    return man2d.store(pDataObject2D)

cdef fillDataObject(CData2D * obj, data):
    if data is None:
        fillDataObjectScalar(obj, 0)
    else:
        if isinstance(data, np.ndarray):
            obj_shape = (obj.getHeight(), obj.getWidth())
            if data.shape != obj_shape:
                raise ValueError(
                  "The dimensions of the data {} do not match those specified "
                  "in the geometry {}".format(data.shape, obj_shape))
            fillDataObjectArray(obj, np.ascontiguousarray(data,dtype=np.float32))
        else:
            fillDataObjectScalar(obj, np.float32(data))

cdef fillDataObjectScalar(CData2D * obj, float s):
    cdef size_t i
    for i in range(obj.getSize()):
        obj.getFloat32Memory()[i] = s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef fillDataObjectArray(CData2D * obj, const float [:,::1] data):
    cdef float [:,::1] cView =  <float[:data.shape[0],:data.shape[1]]> obj.getFloat32Memory()
    cView[:] = data

cdef CData2D * getObject(i) except NULL:
    cdef CData2D * pDataObject = man2d.get(i)
    if pDataObject == NULL:
        raise AstraError("Data object not found")
    if not pDataObject.isInitialized():
        raise AstraError("Data object not initialized properly")
    return pDataObject


def store(i, data):
    cdef CData2D * pDataObject = getObject(i)
    fillDataObject(pDataObject, data)

def get_geometry(i):
    cdef CData2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == TWOPROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        geom = utils.configToDict(pDataObject2.getGeometry().getConfiguration())
    elif pDataObject.getType() == TWOVOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        geom = utils.configToDict(pDataObject3.getGeometry().getConfiguration())
    else:
        raise AstraError("Not a known data object")
    return geom

cdef CProjector2D * getProjector(i) except NULL:
    cdef CProjector2D * proj = manProj.get(i)
    if proj == NULL:
        raise AstraError("Projector not found")
    if not proj.isInitialized():
        raise AstraError("Projector not initialized")
    return proj

def check_compatible(i, proj_id):
    cdef CProjector2D * proj = getProjector(proj_id)
    cdef CData2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == TWOPROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        return pDataObject2.getGeometry().isEqual(proj.getProjectionGeometry())
    elif pDataObject.getType() == TWOVOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        return pDataObject3.getGeometry().isEqual(proj.getVolumeGeometry())
    else:
        raise AstraError("Not a known data object type")

def change_geometry(i, geom):
    cdef XMLConfig *cfg
    cdef CVolumeGeometry2D * pGeometry
    cdef unique_ptr[CProjectionGeometry2D] ppGeometry
    cdef CData2D * pDataObject = getObject(i)
    cdef CFloat32ProjectionData2D * pDataObject2
    cdef CFloat32VolumeData2D * pDataObject3
    if pDataObject.getType() == TWOPROJECTION:
        pDataObject2 = <CFloat32ProjectionData2D * >pDataObject
        cfg = utils.dictToConfig(b'ProjectionGeometry', geom)
        tpe = cfg.self.getAttribute(b'type')
        ppGeometry = constructProjectionGeometry2D(tpe)
        if not ppGeometry:
            raise ValueError("'{}' is not a valid 2D geometry type".format(tpe))
        if not ppGeometry.get().initialize(cfg[0]):
            del cfg
            AstraError('Geometry class could not be initialized', append_log=True)
        geom_shape = (ppGeometry.get().getProjectionAngleCount(), ppGeometry.get().getDetectorCount())
        obj_shape = (pDataObject2.getAngleCount(), pDataObject2.getDetectorCount())
        if geom_shape != obj_shape:
            del cfg
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(obj_shape, geom_shape))
        pDataObject2.changeGeometry(cython.operator.dereference(ppGeometry))
        del cfg
    elif pDataObject.getType() == TWOVOLUME:
        pDataObject3 = <CFloat32VolumeData2D * >pDataObject
        cfg = utils.dictToConfig(b'VolumeGeometry', geom)
        pGeometry = new CVolumeGeometry2D()
        if not pGeometry.initialize(cfg[0]):
            del cfg
            del pGeometry
            raise AstraError('Geometry class could not be initialized', append_log=True)
        geom_shape = (pGeometry.getGridRowCount(), pGeometry.getGridColCount())
        obj_shape = (pDataObject3.getHeight(), pDataObject3.getWidth())
        if geom_shape != obj_shape:
            del cfg
            del pGeometry
            raise ValueError("The dimensions of the data {} do not match those "
                             "specified in the geometry {}".format(obj_shape, geom_shape))
        pDataObject3.changeGeometry(cython.operator.dereference(pGeometry))
        del cfg
        del pGeometry
    else:
        raise AstraError("Not a known data object")

@cython.boundscheck(False)
@cython.wraparound(False)
def get(i):
    cdef CData2D * pDataObject = getObject(i)
    outArr = np.empty((pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef float [:,::1] mView = outArr
    cdef float [:,::1] cView =  <float[:outArr.shape[0],:outArr.shape[1]]> pDataObject.getFloat32Memory()
    mView[:] = cView
    return outArr

def get_shared(i):
    cdef CData2D * pDataObject = getObject(i)
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> pDataObject.getHeight()
    shape[1] = <np.npy_intp> pDataObject.getWidth()
    return np.PyArray_SimpleNewFromData(2,shape,np.NPY_FLOAT32,<void *>pDataObject.getFloat32Memory())


def get_single(i):
    raise NotImplementedError("Not yet implemented")

def shepp_logan(i, modified=True):
    cdef CData2D * pDataObject = getObject(i)
    cdef CFloat32VolumeData2D * pVolumeDataObject = <CFloat32VolumeData2D *>getObject(i)
    generateSheppLogan(pVolumeDataObject, modified);

def info():
    print(wrap_from_bytes(man2d.info()))
