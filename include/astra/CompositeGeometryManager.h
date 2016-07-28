/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#ifndef _INC_ASTRA_COMPOSITEGEOMETRYMANAGER
#define _INC_ASTRA_COMPOSITEGEOMETRYMANAGER

#include "Globals.h"

#ifdef ASTRA_CUDA

#include <list>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>


namespace astra {

class CCompositeVolume;
class CCompositeProjections;
class CFloat32Data3DMemory;
class CFloat32ProjectionData3DMemory;
class CFloat32VolumeData3DMemory;
class CVolumeGeometry3D;
class CProjectionGeometry3D;
class CProjector3D;


struct SGPUParams {
	std::vector<int> GPUIndices;
	size_t memory;
};


class _AstraExport CCompositeGeometryManager {
public:
	CCompositeGeometryManager();

	class CPart;
	typedef std::list<boost::shared_ptr<CPart> > TPartList;
	class CPart {
	public:
		CPart() { }
		CPart(const CPart& other);
		virtual ~CPart() { }

		enum {
			PART_VOL, PART_PROJ
		} eType;

		CFloat32Data3DMemory* pData;
		unsigned int subX;
		unsigned int subY;
		unsigned int subZ;

		bool uploadToGPU();
		bool downloadFromGPU(/*mode?*/);
		virtual void splitX(TPartList& out, size_t maxSize, size_t maxDim, int div) = 0;
		virtual void splitY(TPartList& out, size_t maxSize, size_t maxDim, int div) = 0;
		virtual void splitZ(TPartList& out, size_t maxSize, size_t maxDim, int div) = 0;
		virtual CPart* reduce(const CPart *other) = 0;
		virtual void getDims(size_t &x, size_t &y, size_t &z) = 0;
		size_t getSize();
	};

	class CVolumePart : public CPart {
	public:
		CVolumePart() { eType = PART_VOL; }
		CVolumePart(const CVolumePart& other);
		virtual ~CVolumePart();

		CVolumeGeometry3D* pGeom;

		virtual void splitX(TPartList& out, size_t maxSize, size_t maxDim, int div);
		virtual void splitY(TPartList& out, size_t maxSize, size_t maxDim, int div);
		virtual void splitZ(TPartList& out, size_t maxSize, size_t maxDim, int div);
		virtual CPart* reduce(const CPart *other);
		virtual void getDims(size_t &x, size_t &y, size_t &z);

		CVolumePart* clone() const;
	};
	class CProjectionPart : public CPart {
	public:
		CProjectionPart() { eType = PART_PROJ; }
		CProjectionPart(const CProjectionPart& other);
		virtual ~CProjectionPart();

		CProjectionGeometry3D* pGeom;

		virtual void splitX(TPartList& out, size_t maxSize, size_t maxDim, int div);
		virtual void splitY(TPartList& out, size_t maxSize, size_t maxDim, int div);
		virtual void splitZ(TPartList& out, size_t maxSize, size_t maxDim, int div);
		virtual CPart* reduce(const CPart *other);
		virtual void getDims(size_t &x, size_t &y, size_t &z);

		CProjectionPart* clone() const;
	};

	struct SJob {
	public:
		boost::shared_ptr<CPart> pInput;
		boost::shared_ptr<CPart> pOutput;
		CProjector3D *pProjector; // For a `global' geometry. It will not match
		                          // the geometries of the input and output.


		enum {
			JOB_FP, JOB_BP, JOB_NOP
		} eType;
		enum {
			MODE_ADD, MODE_SET
		} eMode;

	};

	typedef std::list<SJob> TJobList;
	// output part -> list of jobs for that output
	typedef std::map<CPart*, TJobList > TJobSet;

	bool doJobs(TJobList &jobs);

	SJob createJobFP(CProjector3D *pProjector,
                     CFloat32VolumeData3DMemory *pVolData,
                     CFloat32ProjectionData3DMemory *pProjData);
	SJob createJobBP(CProjector3D *pProjector,
                     CFloat32VolumeData3DMemory *pVolData,
                     CFloat32ProjectionData3DMemory *pProjData);

	// Convenience functions for creating and running a single FP or BP job
	bool doFP(CProjector3D *pProjector, CFloat32VolumeData3DMemory *pVolData,
	          CFloat32ProjectionData3DMemory *pProjData);
	bool doBP(CProjector3D *pProjector, CFloat32VolumeData3DMemory *pVolData,
	          CFloat32ProjectionData3DMemory *pProjData);

	bool doFP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3DMemory *>& volData, const std::vector<CFloat32ProjectionData3DMemory *>& projData);
	bool doBP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3DMemory *>& volData, const std::vector<CFloat32ProjectionData3DMemory *>& projData);

	void setGPUIndices(const std::vector<int>& GPUIndices);

	static void setGlobalGPUParams(const SGPUParams& params);

protected:

	bool splitJobs(TJobSet &jobs, size_t maxSize, int div, TJobSet &split);

	std::vector<int> m_GPUIndices;
	size_t m_iMaxSize;


	static SGPUParams* s_params;
};

}

#endif

#endif
