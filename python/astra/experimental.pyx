#-----------------------------------------------------------------------
# Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
#            2014-2015, CWI, Amsterdam
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
#-----------------------------------------------------------------------

# distutils: language = c++
# distutils: libraries = astra

include "config.pxi"

import six
from .PyIncludes cimport *
from libcpp.vector cimport vector

cdef extern from "astra/CompositeGeometryManager.h" namespace "astra":
    cdef cppclass CCompositeGeometryManager:
        bool doFP(CProjector3D *, vector[CFloat32VolumeData3DMemory *], vector[CFloat32ProjectionData3DMemory *])
        bool doBP(CProjector3D *, vector[CFloat32VolumeData3DMemory *], vector[CFloat32ProjectionData3DMemory *])

cdef extern from *:
    CFloat32VolumeData3DMemory * dynamic_cast_vol_mem "dynamic_cast<astra::CFloat32VolumeData3DMemory*>" (CFloat32Data3D * ) except NULL
    CFloat32ProjectionData3DMemory * dynamic_cast_proj_mem "dynamic_cast<astra::CFloat32ProjectionData3DMemory*>" (CFloat32Data3D * ) except NULL

cimport PyProjector3DManager
from .PyProjector3DManager cimport CProjector3DManager
cimport PyData3DManager
from .PyData3DManager cimport CData3DManager

cdef CProjector3DManager * manProj = <CProjector3DManager * >PyProjector3DManager.getSingletonPtr()
cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

def do_composite(projector_id, vol_ids, proj_ids, t):
    cdef vector[CFloat32VolumeData3DMemory *] vol
    cdef CFloat32VolumeData3DMemory * pVolObject
    cdef CFloat32ProjectionData3DMemory * pProjObject
    for v in vol_ids:
        pVolObject = dynamic_cast_vol_mem(man3d.get(v))
        if pVolObject == NULL:
            raise Exception("Data object not found")
        if not pVolObject.isInitialized():
            raise Exception("Data object not initialized properly")
        vol.push_back(pVolObject)
    cdef vector[CFloat32ProjectionData3DMemory *] proj
    for v in proj_ids:
        pProjObject = dynamic_cast_proj_mem(man3d.get(v))
        if pProjObject == NULL:
            raise Exception("Data object not found")
        if not pProjObject.isInitialized():
            raise Exception("Data object not initialized properly")
        proj.push_back(pProjObject)
    cdef CCompositeGeometryManager m
    cdef CProjector3D * projector = manProj.get(projector_id) # may be NULL
    if t == "FP":
        if not m.doFP(projector, vol, proj):
            raise Exception("Failed to perform FP")
    else:
        if not m.doBP(projector, vol, proj):
            raise Exception("Failed to perform BP")

def do_composite_FP(projector_id, vol_ids, proj_ids):
    do_composite(projector_id, vol_ids, proj_ids, "FP")

def do_composite_BP(projector_id, vol_ids, proj_ids):
    do_composite(projector_id, vol_ids, proj_ids, "BP")
