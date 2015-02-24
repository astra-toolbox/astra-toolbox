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
from libcpp cimport bool
from libcpp.string cimport string
from .PyXMLDocument cimport XMLNode

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
		Config()
		XMLNode *self

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


cdef extern from "astra/Float32VolumeData2D.h" namespace "astra":
	cdef cppclass CFloat32VolumeData2D:
		CFloat32VolumeData2D(CVolumeGeometry2D*)
		CVolumeGeometry2D * getGeometry()
		int getWidth()
		int getHeight()
		void changeGeometry(CVolumeGeometry2D*)



cdef extern from "astra/ProjectionGeometry2D.h" namespace "astra":
	cdef cppclass CProjectionGeometry2D:
		CProjectionGeometry2D()
		bool initialize(Config)
		int getDetectorCount()
		int getProjectionAngleCount()
		bool isOfType(string)
		float32 getProjectionAngle(int)
		float32 getDetectorWidth()

cdef extern from "astra/Float32Data2D.h" namespace "astra::CFloat32Data2D":
	cdef enum EDataType:
		BASE,PROJECTION,VOLUME

cdef extern from "astra/Float32Data2D.h" namespace "astra":
	cdef cppclass CFloat32Data2D:
		bool isInitialized()
		int getSize()
		float32 *getData()
		float32 **getData2D()
		int getWidth()
		int getHeight()
		EDataType getType()




cdef extern from "astra/SparseMatrixProjectionGeometry2D.h" namespace "astra":
	cdef cppclass CSparseMatrixProjectionGeometry2D:
		CSparseMatrixProjectionGeometry2D()

cdef extern from "astra/FanFlatProjectionGeometry2D.h" namespace "astra":
	cdef cppclass CFanFlatProjectionGeometry2D:
		CFanFlatProjectionGeometry2D()
		
cdef extern from "astra/FanFlatVecProjectionGeometry2D.h" namespace "astra":
	cdef cppclass CFanFlatVecProjectionGeometry2D:
		CFanFlatVecProjectionGeometry2D()

cdef extern from "astra/ParallelProjectionGeometry2D.h" namespace "astra":
	cdef cppclass CParallelProjectionGeometry2D:
		CParallelProjectionGeometry2D()


cdef extern from "astra/Float32ProjectionData2D.h" namespace "astra":
	cdef cppclass CFloat32ProjectionData2D:
		CFloat32ProjectionData2D(CProjectionGeometry2D*)
		CProjectionGeometry2D * getGeometry()
		void changeGeometry(CProjectionGeometry2D*)
		int getDetectorCount()
		int getAngleCount()

cdef extern from "astra/Algorithm.h" namespace "astra":
	cdef cppclass CAlgorithm:
		bool initialize(Config)
		void run(int)
		bool isInitialized()

cdef extern from "astra/ReconstructionAlgorithm2D.h" namespace "astra":
	cdef cppclass CReconstructionAlgorithm2D:
		bool getResidualNorm(float32&)

cdef extern from "astra/Projector2D.h" namespace "astra":
	cdef cppclass CProjector2D:
		bool isInitialized()
		CProjectionGeometry2D* getProjectionGeometry()
		CVolumeGeometry2D* getVolumeGeometry()
		CSparseMatrix* getMatrix()

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
		
cdef extern from "astra/Float32Data3DMemory.h" namespace "astra":		
	cdef cppclass CFloat32Data3DMemory:
		CFloat32Data3DMemory()
		bool isInitialized()
		int getSize()
		int getWidth()
		int getHeight()
		int getDepth()
		void updateStatistics()
		float32 *getData()
		float32 ***getData3D()


cdef extern from "astra/VolumeGeometry3D.h" namespace "astra":
	cdef cppclass CVolumeGeometry3D:
		CVolumeGeometry3D()
		bool initialize(Config)

cdef extern from "astra/ProjectionGeometry3D.h" namespace "astra":
	cdef cppclass CProjectionGeometry3D:
		CProjectionGeometry3D()
		bool initialize(Config)
		
		
cdef extern from "astra/Float32VolumeData3DMemory.h" namespace "astra":
	cdef cppclass CFloat32VolumeData3DMemory:
		CFloat32VolumeData3DMemory(CVolumeGeometry3D*)


cdef extern from "astra/ParallelProjectionGeometry3D.h" namespace "astra":
	cdef cppclass CParallelProjectionGeometry3D:
		CParallelProjectionGeometry3D()

cdef extern from "astra/ParallelVecProjectionGeometry3D.h" namespace "astra":
	cdef cppclass CParallelVecProjectionGeometry3D:
		CParallelVecProjectionGeometry3D()

cdef extern from "astra/ConeProjectionGeometry3D.h" namespace "astra":
	cdef cppclass CConeProjectionGeometry3D:
		CConeProjectionGeometry3D()
		bool initialize(Config)
		
cdef extern from "astra/ConeVecProjectionGeometry3D.h" namespace "astra":
	cdef cppclass CConeVecProjectionGeometry3D:
		CConeVecProjectionGeometry3D()

cdef extern from "astra/Float32ProjectionData3DMemory.h" namespace "astra":
	cdef cppclass CFloat32ProjectionData3DMemory:
		CFloat32ProjectionData3DMemory(CProjectionGeometry3D*)
		CFloat32ProjectionData3DMemory(CConeProjectionGeometry3D*)		

cdef extern from "astra/Float32Data3D.h" namespace "astra":		
	cdef cppclass CFloat32Data3D:
		CFloat32Data3D()
		bool isInitialized()
		int getSize()
		int getWidth()
		int getHeight()
		int getDepth()
		void updateStatistics()



