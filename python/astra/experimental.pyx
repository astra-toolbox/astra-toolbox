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

include "config.pxi"

IF HAVE_CUDA==True:

    import six
    from .PyIncludes cimport *
    from libcpp.vector cimport vector

    cdef extern from "astra/CompositeGeometryManager.h" namespace "astra::CCompositeGeometryManager::SJob":
        cdef enum EMode:
            MODE_ADD = 0
            MODE_SET = 1
    cdef extern from "astra/CompositeGeometryManager.h" namespace "astra":
        cdef cppclass CCompositeGeometryManager:
            bool doFP(CProjector3D *, vector[CFloat32VolumeData3D *], vector[CFloat32ProjectionData3D *], EMode)
            bool doBP(CProjector3D *, vector[CFloat32VolumeData3D *], vector[CFloat32ProjectionData3D *], EMode)
            bool doFDK(CProjector3D *, CFloat32VolumeData3D *, CFloat32ProjectionData3D *, bool, const float*, EMode)

    cdef extern from *:
        CFloat32VolumeData3D * dynamic_cast_vol_mem "dynamic_cast<astra::CFloat32VolumeData3D*>" (CFloat32Data3D * )
        CFloat32ProjectionData3D * dynamic_cast_proj_mem "dynamic_cast<astra::CFloat32ProjectionData3D*>" (CFloat32Data3D * )

    cdef extern from "astra/Float32ProjectionData3D.h" namespace "astra":
        cdef cppclass CFloat32ProjectionData3D:
            bool isInitialized()
    cdef extern from "astra/Float32VolumeData3D.h" namespace "astra":
        cdef cppclass CFloat32VolumeData3D:
            bool isInitialized()


    cimport PyProjector3DManager
    from .PyProjector3DManager cimport CProjector3DManager
    cimport PyData3DManager
    from .PyData3DManager cimport CData3DManager

    cdef CProjector3DManager * manProj = <CProjector3DManager * >PyProjector3DManager.getSingletonPtr()
    cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

    def do_composite(projector_id, vol_ids, proj_ids, mode, t):
        cdef vector[CFloat32VolumeData3D *] vol
        cdef CFloat32VolumeData3D * pVolObject
        cdef CFloat32ProjectionData3D * pProjObject
        for v in vol_ids:
            pVolObject = dynamic_cast_vol_mem(man3d.get(v))
            if pVolObject == NULL:
                raise Exception("Data object not found")
            if not pVolObject.isInitialized():
                raise Exception("Data object not initialized properly")
            vol.push_back(pVolObject)
        cdef vector[CFloat32ProjectionData3D *] proj
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
            if not m.doFP(projector, vol, proj, mode):
                raise Exception("Failed to perform FP")
        elif t == "BP":
            if not m.doBP(projector, vol, proj, mode):
                raise Exception("Failed to perform BP")
        else:
            raise RuntimeError("internal error: wrong composite op type")

    def do_composite_FP(projector_id, vol_ids, proj_ids):
        do_composite(projector_id, vol_ids, proj_ids, MODE_SET, "FP")

    def do_composite_BP(projector_id, vol_ids, proj_ids):
        do_composite(projector_id, vol_ids, proj_ids, MODE_SET, "BP")

    def accumulate_FP(projector_id, vol_id, proj_id):
        do_composite(projector_id, [vol_id], [proj_id], MODE_ADD, "FP")
    def accumulate_BP(projector_id, vol_id, proj_id):
        do_composite(projector_id, [vol_id], [proj_id], MODE_ADD, "BP")
    def accumulate_FDK(projector_id, vol_id, proj_id):
        cdef CFloat32VolumeData3D * pVolObject
        cdef CFloat32ProjectionData3D * pProjObject
        pVolObject = dynamic_cast_vol_mem(man3d.get(vol_id))
        if pVolObject == NULL:
            raise Exception("Data object not found")
        if not pVolObject.isInitialized():
            raise Exception("Data object not initialized properly")
        pProjObject = dynamic_cast_proj_mem(man3d.get(proj_id))
        if pProjObject == NULL:
            raise Exception("Data object not found")
        if not pProjObject.isInitialized():
            raise Exception("Data object not initialized properly")
        cdef CCompositeGeometryManager m
        cdef CProjector3D * projector = manProj.get(projector_id) # may be NULL
        if not m.doFDK(projector, pVolObject, pProjObject, False, NULL, MODE_ADD):
            raise Exception("Failed to perform FDK")
