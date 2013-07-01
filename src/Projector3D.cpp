/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

#include "astra/Projector3D.h"


namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor
CProjector3D::CProjector3D() : configCheckData(0)
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CProjector3D::~CProjector3D()
{
	if (m_bIsInitialized) clear();
}

//----------------------------------------------------------------------------------------
// Clear for constructors
void CProjector3D::_clear()
{
	m_pProjectionGeometry = NULL;
	m_pVolumeGeometry = NULL;
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Clear
void CProjector3D::clear()
{
	ASTRA_DELETE(m_pProjectionGeometry);
	ASTRA_DELETE(m_pVolumeGeometry);
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check
bool CProjector3D::_check()
{
	// projection geometry
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "Projector3D", "ProjectionGeometry3D not initialized.");
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "Projector3D", "ProjectionGeometry3D not initialized.");

	// volume geometry
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry, "Projector3D", "VolumeGeometry3D not initialized.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "Projector3D", "VolumeGeometry3D not initialized.");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CProjector3D::initialize(const Config& _cfg)
{
	assert(_cfg.self);

	return true;
}

/*
bool CProjector3D::initialize(astra::CProjectionGeometry3D *, astra::CVolumeGeometry3D *)
{
	ASTRA_ASSERT(false);

	return false;
}
*/

//----------------------------------------------------------------------------------------
// Weights of each detector in a projection angle
void CProjector3D::computeProjectionRayWeights(int _iProjection, SPixelWeight* _pfWeightedPixels, int* _piRayStoredPixelCount)
{
	int iPixelBufferSize = getProjectionWeightsCount(_iProjection);
	
	int iDetector = 0;
	for(iDetector = m_pProjectionGeometry->getDetectorTotCount()-1; iDetector >= 0; --iDetector) {
		int iSliceIndex = iDetector / m_pProjectionGeometry->getDetectorColCount(); 
		int iDetectorColIndex = iDetector % m_pProjectionGeometry->getDetectorColCount(); 

		computeSingleRayWeights(_iProjection,									// projector index
								iSliceIndex,									// slice index
								iDetectorColIndex,								// detector index
								&_pfWeightedPixels[iDetector*iPixelBufferSize],	// pixel buffer
								iPixelBufferSize,								// pixel buffer size
								_piRayStoredPixelCount[iDetector]);				// stored pixel count
	}
}
//----------------------------------------------------------------------------------------

} // end namespace
