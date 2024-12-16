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

#include "astra/Projector2D.h"

#include "astra/ProjectionGeometry2DFactory.h"
#include "astra/SparseMatrix.h"

#include "astra/Logging.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// constructor
CProjector2D::CProjector2D() : configCheckData(0)
{

	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// constructor
CProjector2D::CProjector2D(const CProjectionGeometry2D &_pProjectionGeometry, const CVolumeGeometry2D &_pVolumeGeometry) : configCheckData(0)
{
	m_pProjectionGeometry = _pProjectionGeometry.clone();
	m_pVolumeGeometry = _pVolumeGeometry.clone();
	m_bIsInitialized = true;
}

//----------------------------------------------------------------------------------------
// destructor
CProjector2D::~CProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CProjector2D::_clear()
{
	m_pProjectionGeometry = NULL;
	m_pVolumeGeometry = NULL;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CProjector2D::clear()
{
	if (m_pProjectionGeometry) {
		delete m_pProjectionGeometry;
		m_pProjectionGeometry = NULL;
	}
	if (m_pVolumeGeometry) {
		delete m_pVolumeGeometry;
		m_pVolumeGeometry = NULL;
	}
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CProjector2D::_check()
{
	// check pointers
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "Projector2D", "Invalid Projection Geometry Object.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry, "Projector2D", "Invalid Volume Geometry Object.");

	// check initializations
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "Projector2D", "Projection Geometry Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "Projector2D", "Volume Geometry Object Not Initialized.");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CProjector2D::initialize(const Config& _cfg)
{
	ConfigReader<CProjector2D> CR("Projector2D", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	Config *subcfg;
	std::string type;
	bool ok = true;

	ok = CR.getRequiredSubConfig("ProjectionGeometry", subcfg, type);
	if (!ok)
		return false;

	std::unique_ptr<CProjectionGeometry2D> pProjGeometry = constructProjectionGeometry2D(type);
	if (!pProjGeometry) {
		delete subcfg;
		// Invalid geometry type
		ASTRA_CONFIG_CHECK(false, "Projector2D", "Invalid projection geometry type \"%s\" specified.", type.c_str());
	}

	pProjGeometry->initialize(*subcfg);
	delete subcfg;
	m_pProjectionGeometry = pProjGeometry.release();

	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "Projector2D", "ProjectionGeometry not initialized.");	

	ok = CR.getRequiredSubConfig("VolumeGeometry", subcfg, type);
	if (!ok)
		return false;

	m_pVolumeGeometry = new CVolumeGeometry2D();
	m_pVolumeGeometry->initialize(*subcfg);
	delete subcfg;

	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "Projector2D", "VolumeGeometry not initialized.");

	return true;
}

//----------------------------------------------------------------------------------------
// weights of each detector in a projection angle
void CProjector2D::computeProjectionRayWeights(int _iProjection, SPixelWeight* _pfWeightedPixels, int* _piRayStoredPixelCount)
{
	int iPixelBufferSize = getProjectionWeightsCount(_iProjection);
	
	int iDetector;
	for(iDetector = m_pProjectionGeometry->getDetectorCount()-1; iDetector >= 0; --iDetector) {
		computeSingleRayWeights(_iProjection,									// projector index
								iDetector,										// detector index
								&_pfWeightedPixels[iDetector*iPixelBufferSize],	// pixel buffer
								iPixelBufferSize,								// pixel buffer size
								_piRayStoredPixelCount[iDetector]);				// stored pixel count
	}

}

//----------------------------------------------------------------------------------------
// explicit projection matrix
CSparseMatrix* CProjector2D::getMatrix()
{
	unsigned int iProjectionCount = m_pProjectionGeometry->getProjectionAngleCount();
	unsigned int iDetectorCount = m_pProjectionGeometry->getDetectorCount();
	unsigned int iRayCount = iProjectionCount * iDetectorCount;
	unsigned int iVolumeSize = m_pVolumeGeometry->getGridTotCount();
	unsigned long lSize = 0;
	unsigned int iMaxRayLength = 0;
	for (unsigned int i = 0; i < iProjectionCount; ++i) {
		unsigned int iRayLength = getProjectionWeightsCount(i);
		lSize += iDetectorCount * iRayLength;
		if (iRayLength > iMaxRayLength)
			iMaxRayLength = iRayLength;
	}
	CSparseMatrix* pMatrix = new CSparseMatrix(iRayCount, iVolumeSize, lSize);

	if (!pMatrix || !pMatrix->isInitialized()) {
		delete pMatrix;
		return 0;
	}

	SPixelWeight* pEntries = new SPixelWeight[iMaxRayLength];
	unsigned long lMatrixIndex = 0;
	for (unsigned int iRay = 0; iRay < iRayCount; ++iRay) {
		pMatrix->m_plRowStarts[iRay] = lMatrixIndex;
		int iPixelCount;
		int iProjIndex, iDetIndex;
		m_pProjectionGeometry->indexToAngleDetectorIndex(iRay, iProjIndex, iDetIndex);
		computeSingleRayWeights(iProjIndex, iDetIndex, pEntries, iMaxRayLength, iPixelCount);
		
		for (int i = 0; i < iPixelCount; ++i) {
			pMatrix->m_piColIndices[lMatrixIndex] = pEntries[i].m_iIndex;
			pMatrix->m_pfValues[lMatrixIndex] = pEntries[i].m_fWeight;
			++lMatrixIndex;
		}

	}
	pMatrix->m_plRowStarts[iRayCount] = lMatrixIndex;
	
	delete[] pEntries;
	return pMatrix;
}

} // end namespace
