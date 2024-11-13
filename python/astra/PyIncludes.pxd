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

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from .PyXMLDocument cimport XMLNode

include "config.pxi"

cdef extern from "astra/Globals.h" namespace "astra":
    ctypedef float float32
    ctypedef double float64
    ctypedef unsigned short int uint16
    ctypedef signed short int sint16
    ctypedef unsigned char uchar8
    ctypedef signed char schar8
    ctypedef int int32
    ctypedef short int int16

cdef extern from "astra/Config.h" namespace "astra":
    cdef cppclass Config:
        pass

cdef extern from "astra/XMLConfig.h" namespace "astra":
    cdef cppclass XMLConfig(Config):
        XMLConfig(const string& rootname)
        XMLNode self

cdef extern from "astra/VolumeGeometry2D.h" namespace "astra":
    cdef cppclass CVolumeGeometry2D:
        bool initialize(Config)
        int getGridColCount()
        int getGridRowCount()
        int getGridTotCount()
        float32 getWindowLengthX()
        float32 getWindowLengthY()
        float32 getWindowArea()
        float32 getPixelLengthX()
        float32 getPixelLengthY()
        float32 getPixelArea()
        float32 getWindowMinX()
        float32 getWindowMinY()
        float32 getWindowMaxX()
        float32 getWindowMaxY()
        Config* getConfiguration()
        bool isEqual(const CVolumeGeometry2D&)

cdef extern from "astra/Float32Data2D.h" namespace "astra":
    cdef cppclass CCustomMemory[T]:
        pass
    cdef cppclass CFloat32CustomMemory:
        pass

cdef extern from "astra/Float32VolumeData2D.h" namespace "astra":
    cdef cppclass CFloat32VolumeData2D:
        CFloat32VolumeData2D(const CVolumeGeometry2D&)
        CFloat32VolumeData2D(const CVolumeGeometry2D&, CFloat32CustomMemory*)
        CVolumeGeometry2D& getGeometry()
        int getWidth()
        int getHeight()
        void changeGeometry(const CVolumeGeometry2D&)
        Config* getConfiguration()



cdef extern from "astra/ProjectionGeometry2D.h" namespace "astra":
    cdef cppclass CProjectionGeometry2D:
        CProjectionGeometry2D()
        bool initialize(Config)
        int getDetectorCount()
        int getProjectionAngleCount()
        bool isOfType(string)
        float32 getProjectionAngle(int)
        float32 getDetectorWidth()
        Config* getConfiguration()
        bool isEqual(const CProjectionGeometry2D&)

cdef extern from "astra/ProjectionGeometry2DFactory.h" namespace "astra":
    cdef unique_ptr[CProjectionGeometry2D] constructProjectionGeometry2D(const string&type)

cdef extern from "astra/Float32Data2D.h" namespace "astra::CFloat32Data2D":
    cdef enum TWOEDataType "astra::CFloat32Data2D::EDataType":
        TWOPROJECTION "astra::CFloat32Data2D::PROJECTION"
        TWOVOLUME "astra::CFloat32Data2D::VOLUME"

cdef extern from "astra/Data3D.h" namespace "astra::CData3D":
    cdef enum THREEEDataType "astra::CData3D::EDataType":
        THREEPROJECTION "astra::CData3D::PROJECTION"
        THREEVOLUME "astra::CData3D::VOLUME"


cdef extern from "astra/Float32Data2D.h" namespace "astra":
    cdef cppclass CFloat32Data2D:
        bool isInitialized()
        size_t getSize()
        float32 *getData()
        float32 **getData2D()
        int getWidth()
        int getHeight()
        TWOEDataType getType()

cdef extern from "astra/Float32ProjectionData2D.h" namespace "astra":
    cdef cppclass CFloat32ProjectionData2D:
        CFloat32ProjectionData2D(const CProjectionGeometry2D&)
        CFloat32ProjectionData2D(const CProjectionGeometry2D&, CFloat32CustomMemory*)
        CProjectionGeometry2D& getGeometry()
        void changeGeometry(const CProjectionGeometry2D&)
        int getDetectorCount()
        int getAngleCount()

cdef extern from "astra/Algorithm.h" namespace "astra":
    cdef cppclass CAlgorithm:
        bool initialize(Config)
        bool run(int) nogil
        bool isInitialized()

cdef extern from "astra/ReconstructionAlgorithm2D.h" namespace "astra":
    cdef cppclass CReconstructionAlgorithm2D:
        bool getResidualNorm(float32&)

cdef extern from "astra/ReconstructionAlgorithm3D.h" namespace "astra":
    cdef cppclass CReconstructionAlgorithm3D:
        bool getResidualNorm(float32&)

cdef extern from "astra/Projector2D.h" namespace "astra":
    cdef cppclass CProjector2D:
        bool isInitialized()
        bool initialize(Config)
        const CProjectionGeometry2D& getProjectionGeometry()
        const CVolumeGeometry2D& getVolumeGeometry()
        CSparseMatrix* getMatrix()

cdef extern from "astra/Projector3D.h" namespace "astra":
    cdef cppclass CProjector3D:
        bool isInitialized()
        bool initialize(Config)
        const CProjectionGeometry3D& getProjectionGeometry()
        const CVolumeGeometry3D& getVolumeGeometry()

IF HAVE_CUDA==True:
    cdef extern from "astra/CudaProjector3D.h" namespace "astra":
        cdef cppclass CCudaProjector3D

    cdef extern from "astra/CudaProjector2D.h" namespace "astra":
        cdef cppclass CCudaProjector2D

    cdef extern from "astra/Data3D.h" namespace "astraCUDA3d":
        cdef cppclass MemHandle3D:
            pass

    cdef extern from "astra/Data3D.h" namespace "astraCUDA3d":
        cdef MemHandle3D wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch)


cdef extern from "astra/SparseMatrix.h" namespace "astra":
    cdef cppclass CSparseMatrix:
        CSparseMatrix(unsigned int,unsigned int,unsigned long)
        unsigned int m_iWidth
        unsigned int m_iHeight
        unsigned long m_lSize
        bool isInitialized()
        float32* m_pfValues
        unsigned int* m_piColIndices
        unsigned long* m_plRowStarts

cdef extern from "astra/Data3D.h" namespace "astra":
    cdef cppclass CData3D:
        bool isInitialized()
        size_t getSize()
        int getWidth()
        int getHeight()
        int getDepth()
        bool isFloat32Memory()
        float32* getFloat32Memory()
        CDataStorage *getStorage()
        THREEEDataType getType()


#cdef extern from "astra/Data3D.h" namespace "astra":
#    cdef cppclass CFloat32Data3DMemory(CFloat32Data3D):
#        CFloat32Data3DMemory()
#        void updateStatistics()
#        float32 *getData()
#        THREEEDataType getType()


cdef extern from "astra/VolumeGeometry3D.h" namespace "astra":
    cdef cppclass CVolumeGeometry3D:
        CVolumeGeometry3D()
        bool initialize(Config)
        Config * getConfiguration()
        int getGridColCount()
        int getGridRowCount()
        int getGridSliceCount()

cdef extern from "astra/ProjectionGeometry3D.h" namespace "astra":
    cdef cppclass CProjectionGeometry3D:
        CProjectionGeometry3D()
        bool initialize(Config)
        Config * getConfiguration()
        int getProjectionCount()
        int getDetectorColCount()
        int getDetectorRowCount()
        void getProjectedBBox(double, double, double, double, double, double, double&, double&, double&, double&)
        void projectPoint(double, double, double, int, double&, double&)

cdef extern from "astra/ProjectionGeometry3DFactory.h" namespace "astra":
    cdef unique_ptr[CProjectionGeometry3D] constructProjectionGeometry3D(const string&type)

cdef extern from "astra/Data3D.h" namespace "astra":
    cdef cppclass CDataStorage:
        pass
    cdef cppclass CDataMemory[T](CDataStorage):
        CDataMemory(size_t)
        CDataMemory(CCustomMemory[T]*)
        pass

cdef extern from "astra/Data3D.h" namespace "astra":
    cdef cppclass CFloat32VolumeData3D(CData3D):
        CFloat32VolumeData3D(unique_ptr[CVolumeGeometry3D]&&, CDataStorage*)
        CFloat32VolumeData3D(const CVolumeGeometry3D &, CDataStorage*)
        const CVolumeGeometry3D& getGeometry()
        void changeGeometry(const CVolumeGeometry3D&)
        void changeGeometry(unique_ptr[CVolumeGeometry3D] &&)
        int getRowCount()
        int getColCount()
        int getSliceCount()
    cdef CFloat32VolumeData3D* createCFloat32VolumeData3DMemory(unique_ptr[CVolumeGeometry3D]&&)
    cdef CFloat32VolumeData3D* createCFloat32VolumeData3DMemory(const CVolumeGeometry3D &)

cdef extern from "astra/Data3D.h" namespace "astra":
    cdef cppclass CFloat32ProjectionData3D(CData3D):
        CFloat32ProjectionData3D(unique_ptr[CProjectionGeometry3D]&&, CDataStorage*)
        CFloat32ProjectionData3D(const CProjectionGeometry3D &, CDataStorage*)
        const CProjectionGeometry3D& getGeometry()
        void changeGeometry(const CProjectionGeometry3D&)
        void changeGeometry(unique_ptr[CProjectionGeometry3D] &&)
        int getDetectorRowCount()
        int getDetectorColCount()
        int getAngleCount()
    cdef CFloat32ProjectionData3D* createCFloat32ProjectionData3DMemory(unique_ptr[CProjectionGeometry3D]&&)
    cdef CFloat32ProjectionData3D* createCFloat32ProjectionData3DMemory(const CProjectionGeometry3D &)

IF HAVE_CUDA==True:
    cdef extern from "astra/Data3D.h" namespace "astra":
        cdef cppclass CDataGPU(CDataStorage):
            CDataGPU(MemHandle3D)
