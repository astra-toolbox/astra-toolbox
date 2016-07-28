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

#include "astra/FanFlatBeamStripKernelProjector2D.h"

#include <cmath>

#include "astra/DataProjectorPolicies.h"

using namespace std;
using namespace astra;

#include "astra/FanFlatBeamStripKernelProjector2D.inl"

// type of the projector, needed to register with CProjectorFactory
std::string CFanFlatBeamStripKernelProjector2D::type = "strip_fanflat";

//----------------------------------------------------------------------------------------
// default constructor
CFanFlatBeamStripKernelProjector2D::CFanFlatBeamStripKernelProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// constructor
CFanFlatBeamStripKernelProjector2D::CFanFlatBeamStripKernelProjector2D(CFanFlatProjectionGeometry2D* _pProjectionGeometry,
																	   CVolumeGeometry2D* _pReconstructionGeometry)

{
	_clear();
	initialize(_pProjectionGeometry, _pReconstructionGeometry);
}

//----------------------------------------------------------------------------------------
// destructor
CFanFlatBeamStripKernelProjector2D::~CFanFlatBeamStripKernelProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CFanFlatBeamStripKernelProjector2D::_clear()
{
	CProjector2D::_clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CFanFlatBeamStripKernelProjector2D::clear()
{
	CProjector2D::clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CFanFlatBeamStripKernelProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "FanFlatBeamStripKernelProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry), "FanFlatBeamLineKernelProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->getPixelLengthX() == m_pVolumeGeometry->getPixelLengthY(), "FanFlatBeamStripKernelProjector2D", "Pixel height must equal pixel width.");
	
	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CFanFlatBeamStripKernelProjector2D::initialize(const Config& _cfg)
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
bool CFanFlatBeamStripKernelProjector2D::initialize(CFanFlatProjectionGeometry2D* _pProjectionGeometry, 
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
int CFanFlatBeamStripKernelProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	int maxDim = max(m_pVolumeGeometry->getGridRowCount(), m_pVolumeGeometry->getGridColCount());
	return maxDim * 10 + 1;
}

//----------------------------------------------------------------------------------------
// Single Ray Weights
void CFanFlatBeamStripKernelProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
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
std::vector<SDetector2D> CFanFlatBeamStripKernelProjector2D::projectPoint(int _iRow, int _iCol)
{
	//float32 xUL = m_pVolumeGeometry->pixelColToCenterX(_iCol) - m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	//float32 yUL = m_pVolumeGeometry->pixelRowToCenterY(_iRow) - m_pVolumeGeometry->getPixelLengthY() * 0.5f;
	//float32 xUR = m_pVolumeGeometry->pixelColToCenterX(_iCol) + m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	//float32 yUR = m_pVolumeGeometry->pixelRowToCenterY(_iRow) - m_pVolumeGeometry->getPixelLengthY() * 0.5f;
	//float32 xLL = m_pVolumeGeometry->pixelColToCenterX(_iCol) - m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	//float32 yLL = m_pVolumeGeometry->pixelRowToCenterY(_iRow) + m_pVolumeGeometry->getPixelLengthY() * 0.5f;
	//float32 xLR = m_pVolumeGeometry->pixelColToCenterX(_iCol) + m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	//float32 yLR = m_pVolumeGeometry->pixelRowToCenterY(_iRow) + m_pVolumeGeometry->getPixelLengthY() * 0.5f;

	std::vector<SDetector2D> res;
	//// loop projectors and detectors
	//for (int iProjection = 0; iProjection < m_pProjectionGeometry->getProjectionAngleCount(); ++iProjection) {

	//	// get projection angle
	//	float32 theta = m_pProjectionGeometry->getProjectionAngle(iProjection);
	//	if (theta >= 7*PIdiv4) theta -= 2*PI;
	//	bool inverse = false;
	//	if (theta >= 3*PIdiv4) {
	//		theta -= PI;
	//		inverse = true;
	//	}

	//	// calculate distance from the center of the voxel to the ray though the origin
	//	float32 tUL = xUL * cos(theta) + yUL * sin(theta);
	//	float32 tUR = xUR * cos(theta) + yUR * sin(theta);
	//	float32 tLL = xLL * cos(theta) + yLL * sin(theta);
	//	float32 tLR = xLR * cos(theta) + yLR * sin(theta);
	//	if (inverse) {
	//		tUL *= -1.0f;
	//		tUR *= -1.0f;
	//		tLL *= -1.0f;
	//		tLR *= -1.0f;
	//	}
	//	float32 tMin = min(tUL, min(tUR, min(tLL,tLR)));
	//	float32 tMax = max(tUL, max(tUR, max(tLL,tLR)));

	//	// calculate the offset on the detectorarray (in indices)
	//	int dmin = (int)floor(m_pProjectionGeometry->detectorOffsetToIndexFloat(tMin));
	//	int dmax = (int)ceil(m_pProjectionGeometry->detectorOffsetToIndexFloat(tMax));

	//	// add affected detectors to the list
	//	for (int i = dmin; i <= dmax; ++i) {
	//		if (i >= 0 && i < m_pProjectionGeometry->getDetectorCount()) {
	//			SDetector2D det;
	//			det.m_iAngleIndex = iProjection;
	//			det.m_iDetectorIndex = i;
	//			det.m_iIndex = iProjection * getProjectionGeometry()->getDetectorCount() + i;
	//			res.push_back(det);
	//		}
	//	}
	//}

	//// return result vector
	return res;

}

//----------------------------------------------------------------------------------------
