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

#include "astra/FanFlatBeamLineKernelProjector2D.h"

#include <cmath>
#include <cstring>

#include "astra/DataProjectorPolicies.h"

using namespace std;
using namespace astra;

#include "astra/FanFlatBeamLineKernelProjector2D.inl"

// type of the projector, needed to register with CProjectorFactory
std::string CFanFlatBeamLineKernelProjector2D::type = "line_fanflat";


//----------------------------------------------------------------------------------------
// default constructor
CFanFlatBeamLineKernelProjector2D::CFanFlatBeamLineKernelProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// constructor
CFanFlatBeamLineKernelProjector2D::CFanFlatBeamLineKernelProjector2D(CFanFlatProjectionGeometry2D* _pProjectionGeometry,
																	 CVolumeGeometry2D* _pReconstructionGeometry)

{
	_clear();
	initialize(_pProjectionGeometry, _pReconstructionGeometry);
}

//----------------------------------------------------------------------------------------
// destructor
CFanFlatBeamLineKernelProjector2D::~CFanFlatBeamLineKernelProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CFanFlatBeamLineKernelProjector2D::_clear()
{
	CProjector2D::_clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CFanFlatBeamLineKernelProjector2D::clear()
{
	CProjector2D::clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CFanFlatBeamLineKernelProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "FanFlatBeamLineKernelProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry) || dynamic_cast<CFanFlatVecProjectionGeometry2D*>(m_pProjectionGeometry), "FanFlatBeamLineKernelProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->getPixelLengthX() == m_pVolumeGeometry->getPixelLengthY(), "FanFlatBeamLineKernelProjector2D", "Pixel height must equal pixel width.");
	
	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CFanFlatBeamLineKernelProjector2D::initialize(const Config& _cfg)
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
bool CFanFlatBeamLineKernelProjector2D::initialize(CFanFlatProjectionGeometry2D* _pProjectionGeometry, 
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
int CFanFlatBeamLineKernelProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	int maxDim = max(m_pVolumeGeometry->getGridRowCount(), m_pVolumeGeometry->getGridColCount());
	return maxDim * 2 + 1;
}

//----------------------------------------------------------------------------------------
// Single Ray Weights
void CFanFlatBeamLineKernelProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
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
std::vector<SDetector2D> CFanFlatBeamLineKernelProjector2D::projectPoint(int _iRow, int _iCol)
{
	std::vector<SDetector2D> res;
	return res;
}

//----------------------------------------------------------------------------------------
//Result is always in [-PI/2; PI/2]
float32 CFanFlatBeamLineKernelProjector2D::angleBetweenVectors(float32 _fAX, float32 _fAY, float32 _fBX, float32 _fBY)
{
	float32 sinAB = (_fAX*_fBY - _fAY*_fBX)/sqrt((_fAX*_fAX + _fAY*_fAY)*(_fBX*_fBX + _fBY*_fBY));
	return asin(sinAB);
}

//----------------------------------------------------------------------------------------
