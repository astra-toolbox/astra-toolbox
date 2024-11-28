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

#include "astra/ParallelBeamBlobKernelProjector2D.h"

#include <cmath>
#include <algorithm>

#include "astra/DataProjectorPolicies.h"

#include "astra/Logging.h"

using namespace std;
using namespace astra;

#include "astra/ParallelBeamBlobKernelProjector2D.inl"

//----------------------------------------------------------------------------------------
// default constructor
CParallelBeamBlobKernelProjector2D::CParallelBeamBlobKernelProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// constructor
CParallelBeamBlobKernelProjector2D::CParallelBeamBlobKernelProjector2D(const CParallelProjectionGeometry2D &_pProjectionGeometry,
																	   const CVolumeGeometry2D &_pReconstructionGeometry,
																	   float32 _fBlobSize,
																	   float32 _fBlobSampleRate,
																	   int _iBlobSampleCount,
																	   float32* _pfBlobValues)
{
	_clear();
	initialize(_pProjectionGeometry, _pReconstructionGeometry, _fBlobSize, _fBlobSampleRate, _iBlobSampleCount, _pfBlobValues);
}

//----------------------------------------------------------------------------------------
// destructor
CParallelBeamBlobKernelProjector2D::~CParallelBeamBlobKernelProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CParallelBeamBlobKernelProjector2D::_clear()
{
	CProjector2D::_clear();
	m_pfBlobValues.clear();
	m_iBlobSampleCount = 0;
	m_fBlobSize = 0;
	m_fBlobSampleRate = 0;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CParallelBeamBlobKernelProjector2D::clear()
{
	CProjector2D::clear();
	m_pfBlobValues.clear();
	m_iBlobSampleCount = 0;
	m_fBlobSize = 0;
	m_fBlobSampleRate = 0;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CParallelBeamBlobKernelProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "ParallelBeamBlobKernelProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry) || dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry), "ParallelBeamBlobKernelProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(m_iBlobSampleCount > 0, "ParallelBeamBlobKernelProjector2D", "m_iBlobSampleCount should be strictly positive.");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CParallelBeamBlobKernelProjector2D::initialize(const Config& _cfg)
{
	ConfigReader<CProjector2D> CR("ParallelBeamBlobKernelProjector2D", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector2D::initialize(_cfg)) {
		return false;
	}

	Config *subcfg;
	std::string _type;

	if (!CR.getRequiredSubConfig("Kernel", subcfg, _type))
		return false;

	{
		ConfigReader<CProjector2D> SCR("ParallelBeamBlobKernelProjector2D::Kernel", this, *subcfg);

		bool ok = true;
	
		ok &= SCR.getRequiredNumerical("KernelSize", m_fBlobSize);
		ok &= SCR.getRequiredNumerical("SampleRate", m_fBlobSampleRate);
		ok &= SCR.getRequiredInt("SampleCount", m_iBlobSampleCount);
		ok &= SCR.getRequiredNumericalArray("KernelValues", m_pfBlobValues);

		delete subcfg;

		if (!ok)
			return false;
	
		ASTRA_CONFIG_CHECK(m_pfBlobValues.size() == (unsigned int)m_iBlobSampleCount, "BlobProjector", "Number of specified values doesn't match SampleCount.");
	}

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// initialize
bool CParallelBeamBlobKernelProjector2D::initialize(const CParallelProjectionGeometry2D &_pProjectionGeometry,
													const CVolumeGeometry2D &_pVolumeGeometry,
													float32 _fBlobSize,
													float32 _fBlobSampleRate,
													int _iBlobSampleCount,
													float32* _pfBlobValues)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	m_pProjectionGeometry = _pProjectionGeometry.clone();
	m_pVolumeGeometry = _pVolumeGeometry.clone();
	m_fBlobSize = _fBlobSize;
	m_fBlobSampleRate = _fBlobSampleRate;
	m_iBlobSampleCount = _iBlobSampleCount;
	m_pfBlobValues.resize(_iBlobSampleCount);
	for (int i = 0; i <_iBlobSampleCount; i++) {
		m_pfBlobValues[i] = _pfBlobValues[i];
	}
	
	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Get maximum amount of weights on a single ray
int CParallelBeamBlobKernelProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	int maxDim = max(m_pVolumeGeometry->getGridRowCount(), m_pVolumeGeometry->getGridColCount());
	return (int)(maxDim * 2 * (m_fBlobSize+2) + 1);
}
//----------------------------------------------------------------------------------------
// Single Ray Weights
void CParallelBeamBlobKernelProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
														   int _iDetectorIndex, 
														   SPixelWeight* _pWeightedPixels,
														   int _iMaxPixelCount, 
														   int& _iStoredPixelCount)
{
	ASTRA_ASSERT(m_bIsInitialized);
	StorePixelWeightsPolicy p(_pWeightedPixels, _iMaxPixelCount);
	projectSingleRay(_iProjectionIndex, _iDetectorIndex, p);
	_iStoredPixelCount = p.getStoredPixelCount();
}
