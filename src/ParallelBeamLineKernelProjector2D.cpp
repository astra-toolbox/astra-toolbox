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

#include "astra/ParallelBeamLineKernelProjector2D.h"

#include <cmath>
#include <algorithm>

#include "astra/DataProjectorPolicies.h"

#include "astra/Logging.h"

using namespace std;
using namespace astra;

#include "astra/ParallelBeamLineKernelProjector2D.inl"

//----------------------------------------------------------------------------------------
// default constructor
CParallelBeamLineKernelProjector2D::CParallelBeamLineKernelProjector2D()
{

}

//----------------------------------------------------------------------------------------
// constructor
CParallelBeamLineKernelProjector2D::CParallelBeamLineKernelProjector2D(const CParallelProjectionGeometry2D &_pProjectionGeometry,
                                                                       const CVolumeGeometry2D &_pReconstructionGeometry)

{
	initialize(_pProjectionGeometry, _pReconstructionGeometry);
}

//----------------------------------------------------------------------------------------
// destructor
CParallelBeamLineKernelProjector2D::~CParallelBeamLineKernelProjector2D()
{

}

//---------------------------------------------------------------------------------------
// Check
bool CParallelBeamLineKernelProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "ParallelBeamLineKernelProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry) || dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry), "ParallelBeamLineKernelProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(abs(m_pVolumeGeometry->getPixelLengthX() / m_pVolumeGeometry->getPixelLengthY()) - 1 < eps, "ParallelBeamLineKernelProjector2D", "Pixel height must equal pixel width.");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CParallelBeamLineKernelProjector2D::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

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
bool CParallelBeamLineKernelProjector2D::initialize(const CParallelProjectionGeometry2D &_pProjectionGeometry,
                                                    const CVolumeGeometry2D &_pVolumeGeometry)
{
	assert(!m_bIsInitialized);

	// hardcopy geometries
	m_pProjectionGeometry = _pProjectionGeometry.clone();
	m_pVolumeGeometry = _pVolumeGeometry.clone();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Get maximum amount of weights on a single ray
int CParallelBeamLineKernelProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	int maxDim = max(m_pVolumeGeometry->getGridRowCount(), m_pVolumeGeometry->getGridColCount());
	return maxDim * 2 + 1;
}

//----------------------------------------------------------------------------------------
// Single Ray Weights
void CParallelBeamLineKernelProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
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

