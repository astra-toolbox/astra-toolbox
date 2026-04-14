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

include "config.pxi"

from . cimport utils
from .PyIncludes cimport *
from .utils import wrap_from_bytes
from .utils cimport createProjectionGeometry3D
from .utils cimport linkVolFromGeometry2D, linkProjFromGeometry2D
from .log import AstraError

from . cimport PyProjector2DManager
from .PyProjector2DManager cimport CProjector2DManager

cdef extern from "astra/ForwardProjectionAlgorithm.h" namespace "astra":
    cdef cppclass CForwardProjectionAlgorithm(CAlgorithm):
        CForwardProjectionAlgorithm(CProjector2D*, CFloat32VolumeData2D*, CFloat32ProjectionData2D*)
cdef extern from "astra/BackProjectionAlgorithm.h" namespace "astra":
    cdef cppclass CBackProjectionAlgorithm(CReconstructionAlgorithm2D):
        CBackProjectionAlgorithm(CProjector2D*, CFloat32ProjectionData2D*, CFloat32VolumeData2D*)




IF HAVE_CUDA==True:

    from .PyIncludes cimport *
    from libcpp.vector cimport vector

    cdef extern from "astra/Filters.h" namespace "astra":
        cdef enum E_FBPFILTER:
            FILTER_ERROR
            FILTER_NONE
            FILTER_RAMLAK
        cdef cppclass SFilterConfig:
            SFilterConfig()
            E_FBPFILTER m_eType

    cdef extern from "astra/CompositeGeometryManager.h" namespace "astra::CCompositeGeometryManager":
        cdef enum EJobMode:
            MODE_ADD
            MODE_SET
    cdef extern from "astra/CompositeGeometryManager.h" namespace "astra":
        cdef cppclass CCompositeGeometryManager:
            bool doFP(CProjector3D *, vector[CFloat32VolumeData3D *], vector[CFloat32ProjectionData3D *], EJobMode) nogil
            bool doBP(CProjector3D *, vector[CFloat32VolumeData3D *], vector[CFloat32ProjectionData3D *], EJobMode) nogil
            bool doFDK(CProjector3D *, CFloat32VolumeData3D *, CFloat32ProjectionData3D *, bool, SFilterConfig &, EJobMode) nogil

    cdef extern from *:
        CFloat32VolumeData3D * dynamic_cast_vol_mem "dynamic_cast<astra::CFloat32VolumeData3D*>" (CData3D * )
        CFloat32ProjectionData3D * dynamic_cast_proj_mem "dynamic_cast<astra::CFloat32ProjectionData3D*>" (CData3D * )
        CCudaProjector2D* dynamic_cast_cuda_projector "dynamic_cast<astra::CCudaProjector2D*>" (CProjector2D*)

    cdef extern from "astra/CudaForwardProjectionAlgorithm.h" namespace "astra":
        cdef cppclass CCudaForwardProjectionAlgorithm(CAlgorithm):
            CCudaForwardProjectionAlgorithm()
            bool initialize(CProjector2D*, CFloat32VolumeData2D*, CFloat32ProjectionData2D*)
    cdef extern from "astra/CudaBackProjectionAlgorithm.h" namespace "astra":
        cdef cppclass CCudaBackProjectionAlgorithm(CReconstructionAlgorithm2D):
            CCudaBackProjectionAlgorithm()
            bool initialize(CProjector2D*, CFloat32ProjectionData2D*, CFloat32VolumeData2D*)



    from . cimport PyProjector3DManager
    from .PyProjector3DManager cimport CProjector3DManager
    from . cimport PyData3DManager
    from .PyData3DManager cimport CData3DManager
    from . cimport PyData2DManager
    from .PyData2DManager cimport CData2DManager

    cdef CProjector3DManager * manProj3D = <CProjector3DManager * >PyProjector3DManager.getSingletonPtr()
    cdef CData2DManager * man2d = <CData2DManager * >PyData2DManager.getSingletonPtr()
    cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

    def do_composite(projector_id, vol_ids, proj_ids, mode, t):
        if mode != MODE_ADD and mode != MODE_SET:
            raise AstraError("Internal error: wrong composite mode")
        cdef EJobMode eMode = mode;
        cdef vector[CFloat32VolumeData3D *] vol
        cdef CFloat32VolumeData3D * pVolObject
        cdef CFloat32ProjectionData3D * pProjObject
        for v in vol_ids:
            pVolObject = dynamic_cast_vol_mem(man3d.get(v))
            if pVolObject == NULL:
                raise AstraError("Data object not found")
            if not pVolObject.isInitialized():
                raise AstraError("Data object not initialized properly")
            vol.push_back(pVolObject)
        cdef vector[CFloat32ProjectionData3D *] proj
        for v in proj_ids:
            pProjObject = dynamic_cast_proj_mem(man3d.get(v))
            if pProjObject == NULL:
                raise AstraError("Data object not found")
            if not pProjObject.isInitialized():
                raise AstraError("Data object not initialized properly")
            proj.push_back(pProjObject)
        cdef CCompositeGeometryManager m
        cdef CProjector3D * projector = manProj3D.get(projector_id) # may be NULL
        cdef bool ret = True
        if t == "FP":
            with nogil:
                ret = m.doFP(projector, vol, proj, eMode)
            if not ret:
                raise AstraError("Failed to perform FP", append_log=True)
        elif t == "BP":
            with nogil:
                ret = m.doBP(projector, vol, proj, eMode)
            if not ret:
                raise AstraError("Failed to perform BP", append_log=True)
        else:
            raise AstraError("Internal error: wrong composite op type")

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
            raise AstraError("Data object not found")
        if not pVolObject.isInitialized():
            raise AstraError("Data object not initialized properly")
        pProjObject = dynamic_cast_proj_mem(man3d.get(proj_id))
        if pProjObject == NULL:
            raise AstraError("Data object not found")
        if not pProjObject.isInitialized():
            raise AstraError("Data object not initialized properly")
        cdef CCompositeGeometryManager m
        cdef CProjector3D * projector = manProj3D.get(projector_id) # may be NULL
        cdef SFilterConfig filterConfig
        filterConfig.m_eType = FILTER_RAMLAK
        cdef bool ret = True
        with nogil:
            ret = m.doFDK(projector, pVolObject, pProjObject, False, filterConfig, MODE_ADD)
        if not ret:
            raise AstraError("Failed to perform FDK", append_log=True)

    from . cimport utils
    from .utils cimport linkVolFromGeometry3D, linkProjFromGeometry3D

    def direct_FPBP3D(projector_id, vol, proj, mode, t):
        if mode != MODE_ADD and mode != MODE_SET:
            raise AstraError("Internal error: wrong composite mode")
        cdef EJobMode eMode = mode
        cdef CProjector3D * projector = manProj3D.get(projector_id)
        if projector == NULL:
            raise AstraError("Projector not found")
        cdef CFloat32VolumeData3D * pVol = linkVolFromGeometry3D(projector.getVolumeGeometry(), vol)
        cdef CFloat32ProjectionData3D * pProj = linkProjFromGeometry3D(projector.getProjectionGeometry(), proj)
        cdef vector[CFloat32VolumeData3D *] vols
        cdef vector[CFloat32ProjectionData3D *] projs
        vols.push_back(pVol)
        projs.push_back(pProj)
        cdef CCompositeGeometryManager m
        cdef bool ret = True
        try:
            if t == "FP":
                with nogil:
                    ret = m.doFP(projector, vols, projs, eMode)
                if not ret:
                    AstraError("Failed to perform FP", append_log=True)
            elif t == "BP":
                with nogil:
                    ret = m.doBP(projector, vols, projs, eMode)
                if not ret:
                    AstraError("Failed to perform BP", append_log=True)
            else:
                AstraError("Internal error: wrong op type")
        finally:
            del pVol
            del pProj

    def direct_FP3D(projector_id, vol, proj):
        """Perform a 3D forward projection with pre-allocated input/output.

        :param projector_id: A 3D projector object handle
        :type datatype: :class:`int`
        :param vol: The input data, as either a numpy array, or other DLPack object
        :type datatype: :class:`numpy.ndarray`
        :param proj: The pre-allocated output data, either numpy array, or other DLPack object
        :type datatype: :class:`numpy.ndarray`
        """
        direct_FPBP3D(projector_id, vol, proj, MODE_SET, "FP")

    def direct_BP3D(projector_id, vol, proj):
        """Perform a 3D back projection with pre-allocated input/output.

        :param projector_id: A 3D projector object handle
        :type datatype: :class:`int`
        :param vol: The pre-allocated output data, as either a numpy array, or other DLPack object
        :type datatype: :class:`numpy.ndarray`
        :param proj: The input data, either numpy array, or other DLPack objects
        :type datatype: :class:`numpy.ndarray`
        """
        direct_FPBP3D(projector_id, vol, proj, MODE_SET, "BP")

    def getProjectedBBox(geometry, minx, maxx, miny, maxy, minz, maxz):
        cdef unique_ptr[CProjectionGeometry3D] ppGeometry
        cdef double minu=0., maxu=0., minv=0., maxv=0.
        ppGeometry = createProjectionGeometry3D(geometry)
        ppGeometry.get().getProjectedBBox(minx, maxx, miny, maxy, minz, maxz, minu, maxu, minv, maxv)
        return (minv, maxv)

    def projectPoint(geometry, x, y, z, angle):
        cdef unique_ptr[CProjectionGeometry3D] ppGeometry
        cdef double u=0., v=0.
        ppGeometry = createProjectionGeometry3D(geometry)
        ppGeometry.get().projectPoint(x, y, z, angle, u, v)
        return (u, v)

    def direct_FPBP2D(projector_id, vol, proj, t):
        cdef CProjector2DManager * manProj2D = <CProjector2DManager * >PyProjector2DManager.getSingletonPtr()
        cdef CProjector2D * projector = manProj2D.get(projector_id)
        if projector == NULL:
            raise AstraError("Projector not found")
        cdef CFloat32VolumeData2D * pVol = linkVolFromGeometry2D(projector.getVolumeGeometry(), vol)
        cdef CFloat32ProjectionData2D * pProj = linkProjFromGeometry2D(projector.getProjectionGeometry(), proj)
        cdef CAlgorithm *pAlg = NULL
        cdef CCudaProjector2D *pCudaProj = dynamic_cast_cuda_projector(projector)
        cdef bool ret = True
        cdef CCudaForwardProjectionAlgorithm * pf
        cdef CCudaBackProjectionAlgorithm * pb

        try:
            if pCudaProj == NULL:
                if t == "FP":
                    pAlg = new CForwardProjectionAlgorithm(projector, pVol, pProj)
                elif t == "BP":
                    pAlg = new CBackProjectionAlgorithm(projector, pProj, pVol)
                else:
                    raise AstraError("Internal error: wrong op type")
            else:
                if t == "FP":
                    pf = new CCudaForwardProjectionAlgorithm()
                    pf.initialize(projector, pVol, pProj)
                    pAlg = pf
                elif t == "BP":
                    pb = new CCudaBackProjectionAlgorithm()
                    pb.initialize(projector, pProj, pVol)
                    pAlg = pb
                else:
                    raise AstraError("Internal error: wrong op type")

            if not pAlg.isInitialized():
                raise AstraError("Failed to initialize algorithm")

            with nogil:
                pAlg.run(1)
        finally:
            del pVol
            del pProj
            del pAlg

ELSE:

    def direct_FPBP2D(projector_id, vol, proj, t):
        cdef CProjector2DManager * manProj2D = <CProjector2DManager * >PyProjector2DManager.getSingletonPtr()
        cdef CProjector2D * projector = manProj2D.get(projector_id)
        if projector == NULL:
            raise AstraError("Projector not found")
        cdef CFloat32VolumeData2D * pVol = linkVolFromGeometry2D(projector.getVolumeGeometry(), vol)
        cdef CFloat32ProjectionData2D * pProj = linkProjFromGeometry2D(projector.getProjectionGeometry(), proj)
        cdef CAlgorithm *pAlg = NULL
        cdef bool ret = True

        try:
            if t == "FP":
                pAlg = new CForwardProjectionAlgorithm(projector, pVol, pProj)
            elif t == "BP":
                pAlg = new CBackProjectionAlgorithm(projector, pProj, pVol)
            else:
                raise AstraError("Internal error: wrong op type")

            if not pAlg.isInitialized():
                raise AstraError("Failed to initialize algorithm")

            with nogil:
                pAlg.run(1)
        finally:
            del pVol
            del pProj
            del pAlg




def direct_FP2D(projector_id, vol, proj):
    """Perform a 2D forward projection with pre-allocated input/output.

    :param projector_id: A 2D projector object handle
    :type datatype: :class:`int`
    :param vol: The input data, as either a numpy array, or other DLPack object
    :type datatype: :class:`numpy.ndarray`
    :param proj: The pre-allocated output data, either numpy array, or other DLPack object
    :type datatype: :class:`numpy.ndarray`
    """
    direct_FPBP2D(projector_id, vol, proj, "FP")

def direct_BP2D(projector_id, vol, proj):
    """Perform a 2D back projection with pre-allocated input/output.

    :param projector_id: A 2D projector object handle
    :type datatype: :class:`int`
    :param vol: The pre-allocated output data, as either a numpy array, or other DLPack object
    :type datatype: :class:`numpy.ndarray`
    :param proj: The input data, either numpy array, or other DLPack objects
    :type datatype: :class:`numpy.ndarray`
    """
    direct_FPBP2D(projector_id, vol, proj, "BP")


