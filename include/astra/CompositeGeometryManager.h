/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

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

#include "Filters.h"

#ifdef ASTRA_CUDA

#include <list>
#include <map>
#include <vector>
#include <memory>


namespace astra {

class CCompositeVolume;
class CCompositeProjections;
class CData3D;
class CFloat32ProjectionData3D;
class CFloat32VolumeData3D;
class CVolumeGeometry3D;
class CProjectionGeometry3D;
class CProjector3D;


struct SGPUParams {
	std::vector<int> GPUIndices;
	size_t memory;
};

struct SFDKSettings {
	bool bShortScan;
	SFilterConfig filterConfig;
};


class _AstraExport CCompositeGeometryManager {
public:
	CCompositeGeometryManager();

	class CPart;
	typedef std::list<std::unique_ptr<CPart> > TPartList;
	class CPart {
	public:
		CPart() : eType(PART_INVALID), pData(0), subX(0), subY(0), subZ(0) { }
		CPart(const CPart& other);
		virtual ~CPart() { }

		enum {
			PART_INVALID, PART_VOL, PART_PROJ
		} eType;

		CData3D* pData;
		unsigned int subX;
		unsigned int subY;
		unsigned int subZ;

		bool uploadToGPU();
		bool downloadFromGPU(/*mode?*/);
		virtual void getDims(size_t &x, size_t &y, size_t &z) const = 0;
		size_t getSize() const;

		bool canSplitAndReduce() const;
		bool isFull() const;
	};

	class CVolumePart : public CPart {
	public:
		CVolumePart() { eType = PART_VOL; }
		CVolumePart(const CVolumePart& other);
		CVolumePart(CFloat32VolumeData3D *pVolData);

		virtual ~CVolumePart();

		CVolumeGeometry3D* pGeom;

		virtual void getDims(size_t &x, size_t &y, size_t &z) const;

		CVolumePart* clone() const;
	};
	class CProjectionPart : public CPart {
	public:
		CProjectionPart() { eType = PART_PROJ; }
		CProjectionPart(const CProjectionPart& other);
		CProjectionPart(CFloat32ProjectionData3D *pProjData);
		virtual ~CProjectionPart();

		CProjectionGeometry3D* pGeom;

		virtual void getDims(size_t &x, size_t &y, size_t &z) const;

		CProjectionPart* clone() const;
	};

	enum EJobType {
		JOB_FP, JOB_BP, JOB_FDK, JOB_NOP
	};
	enum EJobMode {
		MODE_ADD = 0,
		MODE_SET = 1
	};

	struct SJob {
	public:
		SJob(std::unique_ptr<CPart> &&_pInput, std::unique_ptr<CPart> &&_pOutput)
			: pInput(std::move(_pInput)), pOutput(std::move(_pOutput)), pProjector(0), FDKSettings{} { }
		std::unique_ptr<CPart> pInput;
		std::unique_ptr<CPart> pOutput;
		CProjector3D *pProjector; // For a `global' geometry. It will not match
		                          // the geometries of the input and output.

		SFDKSettings FDKSettings;

		EJobType eType;
		EJobMode eMode;
	};

	// job structure that is lacking the output part pointer
	struct SJobInternal {
	public:
		SJobInternal(std::unique_ptr<CPart> &&_pInput)
			: pInput(std::move(_pInput)), pProjector(0), FDKSettings{} { }
		std::unique_ptr<CPart> pInput;
		CProjector3D *pProjector;
		SFDKSettings FDKSettings;
		EJobType eType;
		EJobMode eMode;
	};

	typedef std::list<SJob> TJobList;
	typedef std::list<SJobInternal> TJobListInternal;
	// pairs of (output part, list of jobs for that output)
	typedef std::list<std::pair<std::unique_ptr<CPart>, TJobListInternal>> TJobSetInternal;

	// Perform a list of jobs. The outputs are assumed to be disjoint.
	bool doJobs(TJobList &jobs);

	// Perform a list of jobs. The outputs are assumed to be disjoint.
	bool doJobs(TJobSetInternal &jobs);

	SJob createJobFP(CProjector3D *pProjector,
                     CFloat32VolumeData3D *pVolData,
                     CFloat32ProjectionData3D *pProjData,
	             EJobMode eMode);
	SJob createJobBP(CProjector3D *pProjector,
                     CFloat32VolumeData3D *pVolData,
                     CFloat32ProjectionData3D *pProjData,
	             EJobMode eMode);

	// Convenience functions for creating and running a single FP or BP job
	bool doFP(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
	          CFloat32ProjectionData3D *pProjData, EJobMode eMode = MODE_SET);
	bool doBP(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
	          CFloat32ProjectionData3D *pProjData, EJobMode eMode = MODE_SET);
	bool doFDK(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
	          CFloat32ProjectionData3D *pProjData, bool bShortScan,
	          const SFilterConfig &filterConfig, EJobMode eMode = MODE_SET);

	bool doFP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3D *>& volData, const std::vector<CFloat32ProjectionData3D *>& projData, EJobMode eMode = MODE_SET);
	bool doBP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3D *>& volData, const std::vector<CFloat32ProjectionData3D *>& projData, EJobMode eMode = MODE_SET);

	void setGPUIndices(const std::vector<int>& GPUIndices);

	static void setGlobalGPUParams(const SGPUParams& params);

protected:

	bool splitJobs(TJobSetInternal &jobs, size_t maxSize, int div, TJobSetInternal &split);

	std::vector<int> m_GPUIndices;
	size_t m_iMaxSize;


	static SGPUParams* s_params;
};

}

#endif

#endif
