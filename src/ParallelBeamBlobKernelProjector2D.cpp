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
$Id$
*/

#include "astra/ParallelBeamBlobKernelProjector2D.h"

#include <cmath>

#include "astra/DataProjectorPolicies.h"

using namespace std;
using namespace astra;

#include "astra/ParallelBeamBlobKernelProjector2D.inl"

// type of the projector, needed to register with CProjectorFactory
std::string CParallelBeamBlobKernelProjector2D::type = "blob";

//----------------------------------------------------------------------------------------
// default constructor
CParallelBeamBlobKernelProjector2D::CParallelBeamBlobKernelProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// constructor
CParallelBeamBlobKernelProjector2D::CParallelBeamBlobKernelProjector2D(CParallelProjectionGeometry2D* _pProjectionGeometry, 
																	   CVolumeGeometry2D* _pReconstructionGeometry,
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
	m_pfBlobValues = NULL;
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
	if (m_pfBlobValues) {
		delete[] m_pfBlobValues;
		m_pfBlobValues = NULL;
	}
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

	ASTRA_CONFIG_CHECK(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry), "ParallelBeamBlobKernelProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(m_iBlobSampleCount > 0, "ParallelBeamBlobKernelProjector2D", "m_iBlobSampleCount should be strictly positive.");
	ASTRA_CONFIG_CHECK(m_pfBlobValues, "ParallelBeamBlobKernelProjector2D", "Invalid Volume Geometry Object.");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CParallelBeamBlobKernelProjector2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector2D::initialize(_cfg)) {
		return false;
	}

	// required: Kernel
	XMLNode node = _cfg.self.getSingleNode("Kernel");
	ASTRA_CONFIG_CHECK(node, "BlobProjector", "No Kernel tag specified.");
	{
		// Required: KernelSize
		XMLNode node2 = node.getSingleNode("KernelSize");
		ASTRA_CONFIG_CHECK(node2, "BlobProjector", "No Kernel/KernelSize tag specified.");
		m_fBlobSize = node2.getContentNumerical();

		// Required: SampleRate
		node2 = node.getSingleNode("SampleRate");
		ASTRA_CONFIG_CHECK(node2, "BlobProjector", "No Kernel/SampleRate tag specified.");
		m_fBlobSampleRate = node2.getContentNumerical();
	
		// Required: SampleCount
		node2 = node.getSingleNode("SampleCount");
		ASTRA_CONFIG_CHECK(node2, "BlobProjector", "No Kernel/SampleCount tag specified.");
		m_iBlobSampleCount = node2.getContentInt();
	
		// Required: KernelValues
		node2 = node.getSingleNode("KernelValues");
		ASTRA_CONFIG_CHECK(node2, "BlobProjector", "No Kernel/KernelValues tag specified.");
		vector<float32> values = node2.getContentNumericalArray();
		ASTRA_CONFIG_CHECK(values.size() == (unsigned int)m_iBlobSampleCount, "BlobProjector", "Number of specified values doesn't match SampleCount.");
		m_pfBlobValues = new float32[m_iBlobSampleCount];
		for (int i = 0; i < m_iBlobSampleCount; i++) {
			m_pfBlobValues[i] = values[i];
		}

		// Required: KernelValues
		node2 = node.getSingleNode("KernelValuesNeg");
		ASTRA_CONFIG_CHECK(node2, "BlobProjector", "No Kernel/KernelValuesNeg tag specified.");
		vector<float32> values2 = node2.getContentNumericalArray();
		ASTRA_CONFIG_CHECK(values2.size() == (unsigned int)m_iBlobSampleCount, "BlobProjector", "Number of specified values doesn't match SampleCount.");
		m_pfBlobValuesNeg = new float32[m_iBlobSampleCount];
		for (int i = 0; i < m_iBlobSampleCount; i++) {
			m_pfBlobValuesNeg[i] = values2[i];
		}

	}

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// initialize
bool CParallelBeamBlobKernelProjector2D::initialize(CParallelProjectionGeometry2D* _pProjectionGeometry, 
													CVolumeGeometry2D* _pVolumeGeometry,
													float32 _fBlobSize,
													float32 _fBlobSampleRate,
													int _iBlobSampleCount,
													float32* _pfBlobValues)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	ASTRA_CONFIG_CHECK(_pProjectionGeometry, "BlobProjector", "Invalid ProjectionGeometry Object");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry, "BlobProjector", "Invalid ProjectionGeometry Object");
	m_pProjectionGeometry = _pProjectionGeometry->clone();
	m_pVolumeGeometry = _pVolumeGeometry->clone();
	m_fBlobSize = _fBlobSize;
	m_fBlobSampleRate = _fBlobSampleRate;
	m_iBlobSampleCount = _iBlobSampleCount;
	m_pfBlobValues = new float32[_iBlobSampleCount];
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
//----------------------------------------------------------------------------------------
// Splat a single point
std::vector<SDetector2D> CParallelBeamBlobKernelProjector2D::projectPoint(int _iRow, int _iCol)
{
	float32 x = m_pVolumeGeometry->pixelColToCenterX(_iCol);
	float32 y = m_pVolumeGeometry->pixelRowToCenterY(_iRow);

	std::vector<SDetector2D> res;
	// loop projectors and detectors
	for (int iProjection = 0; iProjection < m_pProjectionGeometry->getProjectionAngleCount(); ++iProjection) {

		// get projection angle
		float32 theta = m_pProjectionGeometry->getProjectionAngle(iProjection);
		if (theta >= 7*PIdiv4) theta -= 2*PI;
		bool inverse = false;
		if (theta >= 3*PIdiv4) {
			theta -= PI;
			inverse = true;
		}

		// calculate distance from the center of the voxel to the ray though the origin
		float32 t = x * cos(theta) + y * sin(theta);
		if (inverse) t *= -1.0f;

		// calculate the offset on the detectorarray (in indices)
		float32 d = m_pProjectionGeometry->detectorOffsetToIndexFloat(t);
		int dmin = (int)ceil(d - m_fBlobSize);
		int dmax = (int)floor(d + m_fBlobSize);

		// add affected detectors to the list
		for (int i = dmin; i <= dmax; ++i) {
			if (d >= 0 && d < m_pProjectionGeometry->getDetectorCount()) {
				SDetector2D det;
				det.m_iAngleIndex = iProjection;
				det.m_iDetectorIndex = i;
				det.m_iIndex = iProjection * getProjectionGeometry()->getDetectorCount() + i;
				res.push_back(det);
			}
		}
	}

	// return result vector
	return res;

}
