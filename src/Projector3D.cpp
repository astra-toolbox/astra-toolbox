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

#include "astra/Projector3D.h"

#include "astra/VolumeGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"


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
	ConfigStackCheck<CProjector3D> CC("Projector3D", this, _cfg);

	XMLNode node;

	node = _cfg.self.getSingleNode("ProjectionGeometry");
	ASTRA_CONFIG_CHECK(node, "Projector3D", "No ProjectionGeometry tag specified.");
	std::string type = node.getAttribute("type");
	CProjectionGeometry3D* pProjGeometry = 0;
	if (type == "parallel3d") {
		pProjGeometry = new CParallelProjectionGeometry3D();
	} else if (type == "parallel3d_vec") {
		pProjGeometry = new CParallelVecProjectionGeometry3D();
	} else if (type == "cone") {
		pProjGeometry = new CConeProjectionGeometry3D();
	} else if (type == "cone_vec") {
		pProjGeometry = new CConeVecProjectionGeometry3D();
	} else {
		// Invalid geometry type
		ASTRA_CONFIG_CHECK(false, "Projector3D", "Invalid projection geometry type specified.");
	}
	pProjGeometry->initialize(Config(node)); // this deletes node
	m_pProjectionGeometry = pProjGeometry;
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "Projector3D", "ProjectionGeometry not initialized.");
	CC.markNodeParsed("ProjectionGeometry");

	node = _cfg.self.getSingleNode("VolumeGeometry");
	ASTRA_CONFIG_CHECK(node, "Projector3D", "No VolumeGeometry tag specified.");
	CVolumeGeometry3D* pVolGeometry = new CVolumeGeometry3D();
	pVolGeometry->initialize(Config(node)); // this deletes node
	m_pVolumeGeometry = pVolGeometry;
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "Projector3D", "VolumeGeometry not initialized.");
	CC.markNodeParsed("VolumeGeometry");

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
