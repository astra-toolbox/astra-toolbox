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

#include "astra/Projector3D.h"

#include "astra/VolumeGeometry3D.h"
#include "astra/ProjectionGeometry3D.h"
#include "astra/ProjectionGeometry3DFactory.h"

#include "astra/Logging.h"

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
	m_pProjectionGeometry.reset();
	m_pVolumeGeometry .reset();
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Clear
void CProjector3D::clear()
{
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
	ConfigReader<CProjector3D> CR("Projector3D", this, _cfg);

	Config *subcfg;
	std::string type;
	bool ok = true;

	ok = CR.getRequiredSubConfig("ProjectionGeometry", subcfg, type);
	if (!ok)
		return false;

	std::unique_ptr<CProjectionGeometry3D> pProjGeometry = constructProjectionGeometry3D(type);

	if (!pProjGeometry) {
		delete subcfg;
		// Invalid geometry type
		ASTRA_CONFIG_CHECK(false, "Projector3D", "Invalid projection geometry type \"%s\" specified.", type.c_str());
	}
	pProjGeometry->initialize(*subcfg);
	delete subcfg;

	m_pProjectionGeometry = std::move(pProjGeometry);
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "Projector3D", "ProjectionGeometry not initialized.");

	ok = CR.getRequiredSubConfig("VolumeGeometry", subcfg, type);
	if (!ok)
		return false;

	m_pVolumeGeometry = std::make_unique<CVolumeGeometry3D>();
	m_pVolumeGeometry->initialize(*subcfg);
	delete subcfg;
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "Projector3D", "VolumeGeometry not initialized.");

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
