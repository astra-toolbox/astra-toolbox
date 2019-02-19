/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#include "astra/ParallelBeamDistanceDrivenProjector2D.h"

#include <cmath>
#include <algorithm>

#include "astra/DataProjectorPolicies.h"

namespace astra {

#include "astra/ParallelBeamDistanceDrivenProjector2D.inl"

// type of the projector, needed to register with CProjectorFactory
std::string CParallelBeamDistanceDrivenProjector2D::type = "distance_driven";

//----------------------------------------------------------------------------------------
// default constructor
CParallelBeamDistanceDrivenProjector2D::CParallelBeamDistanceDrivenProjector2D()
{
	_clear();
}


//----------------------------------------------------------------------------------------
// constructor
CParallelBeamDistanceDrivenProjector2D::CParallelBeamDistanceDrivenProjector2D(CParallelProjectionGeometry2D* _pProjectionGeometry,
																		   CVolumeGeometry2D* _pReconstructionGeometry)

{
	_clear();
	initialize(_pProjectionGeometry, _pReconstructionGeometry);
}

//----------------------------------------------------------------------------------------
// destructor
CParallelBeamDistanceDrivenProjector2D::~CParallelBeamDistanceDrivenProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - CParallelBeamDistanceDrivenProjector2D
void CParallelBeamDistanceDrivenProjector2D::_clear()
{
	CProjector2D::_clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CParallelBeamDistanceDrivenProjector2D::clear()
{
	CProjector2D::clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CParallelBeamDistanceDrivenProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "ParallelBeamDistanceDrivenProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry) || dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry), "ParallelBeamDistanceDrivenProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(abs(m_pVolumeGeometry->getPixelLengthX() / m_pVolumeGeometry->getPixelLengthY()) - 1 < eps, "ParallelBeamDistanceDrivenProjector2D", "Pixel height must equal pixel width.");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CParallelBeamDistanceDrivenProjector2D::initialize(const Config& _cfg)
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

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize
bool CParallelBeamDistanceDrivenProjector2D::initialize(CParallelProjectionGeometry2D* _pProjectionGeometry, 
                                                        CVolumeGeometry2D* _pVolumeGeometry)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// hardcopy geometries
	m_pProjectionGeometry = _pProjectionGeometry->clone();
	m_pVolumeGeometry = _pVolumeGeometry->clone();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Get maximum amount of weights on a single ray
int CParallelBeamDistanceDrivenProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	int maxDim = std::max(m_pVolumeGeometry->getGridRowCount(), m_pVolumeGeometry->getGridColCount());
	int scale = m_pProjectionGeometry->getDetectorWidth() / std::min(m_pVolumeGeometry->getPixelLengthX(), m_pVolumeGeometry->getPixelLengthY());
	return maxDim * scale * 10 + 1;
}

//----------------------------------------------------------------------------------------
// Single Ray Weights
void CParallelBeamDistanceDrivenProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
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

}
