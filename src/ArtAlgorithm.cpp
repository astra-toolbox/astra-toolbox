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

#include "astra/ArtAlgorithm.h"

#include "astra/AstraObjectManager.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CArtAlgorithm::CArtAlgorithm() 
 : CReconstructionAlgorithm2D()
{
	m_fLambda = 1.0f;
	m_iCurrentRay = 0;
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CArtAlgorithm::~CArtAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CArtAlgorithm::_clear()
{
	CReconstructionAlgorithm2D::_clear();
	m_piDetectorOrder.clear();
	m_piProjectionOrder.clear();
	m_iCurrentRay = 0;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CArtAlgorithm::clear()
{
	CReconstructionAlgorithm2D::clear();
	m_piDetectorOrder.clear();
	m_piProjectionOrder.clear();
	m_fLambda = 1.0f;
	m_iCurrentRay = 0;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CArtAlgorithm::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm2D::_check(), "ART", "Error in ReconstructionAlgorithm2D initialization");

	ASTRA_CONFIG_CHECK(m_piProjectionOrder.size() == m_piDetectorOrder.size(), "ART", "Different sizes of ray order lists.");

	// check ray order list
	for (unsigned int i = 0; i < m_piProjectionOrder.size(); i++) {
		if (m_piProjectionOrder[i] < 0 || m_piProjectionOrder[i] > m_pSinogram->getAngleCount()-1) {
			ASTRA_CONFIG_CHECK(false, "ART", "Invalid value in ray order list.");
		}
		if (m_piDetectorOrder[i] < 0 || m_piDetectorOrder[i] > m_pSinogram->getDetectorCount()-1) {
			ASTRA_CONFIG_CHECK(false, "ART", "Invalid value in ray order list.");
		}
	}

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CArtAlgorithm::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("ArtAlgorithm", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}
	
	// initialization of parent class
	if (!CReconstructionAlgorithm2D::initialize(_cfg)) {
		return false;
	}

	// ray order
	std::string projOrder;
	if (!CR.getOptionString("RayOrder", projOrder, "sequential"))
		return false;
	m_iCurrentRay = 0;
	int iRayCount = m_pProjector->getProjectionGeometry().getProjectionAngleCount() *
		m_pProjector->getProjectionGeometry().getDetectorCount();
	if (projOrder == "sequential") {
		m_piProjectionOrder.resize(iRayCount);
		m_piDetectorOrder.resize(iRayCount);
		for (int i = 0; i < iRayCount; i++) {
			m_piProjectionOrder[i] = i / m_pProjector->getProjectionGeometry().getDetectorCount();
			m_piDetectorOrder[i] = i % m_pProjector->getProjectionGeometry().getDetectorCount();
		}
	} else if (projOrder == "custom") {
		std::vector<int> rayOrderList;
		if (!CR.getOptionIntArray("RayOrderList", rayOrderList))
			return false;
		iRayCount = rayOrderList.size() / 2;
		m_piProjectionOrder.resize(iRayCount);
		m_piDetectorOrder.resize(iRayCount);
		for (int i = 0; i < iRayCount; i++) {
			m_piProjectionOrder[i] = rayOrderList[2*i];
			m_piDetectorOrder[i] = rayOrderList[2*i+1];
		}
	} else {
		ASTRA_ERROR("Unknown RayOrder");
		return false;
	}

	bool ok = true;

	// "Lambda" is replaced by the more descriptive "Relaxation"
	if (CR.hasOption("Relaxation"))
		ok &= CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f);
	else
		ok &= CR.getOptionNumerical("Lambda", m_fLambda, 1.0f);

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CArtAlgorithm::initialize(CProjector2D* _pProjector, 
							   CFloat32ProjectionData2D* _pSinogram, 
							   CFloat32VolumeData2D* _pReconstruction)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	// ray order
	m_iCurrentRay = 0;
	int iRayCount = _pProjector->getProjectionGeometry().getDetectorCount() *
		_pProjector->getProjectionGeometry().getProjectionAngleCount();
	m_piProjectionOrder.resize(iRayCount);
	m_piDetectorOrder.resize(iRayCount);
	for (int i = 0; i < iRayCount; i++) {
		m_piProjectionOrder[i] = i / _pProjector->getProjectionGeometry().getDetectorCount();
		m_piDetectorOrder[i] = i % _pProjector->getProjectionGeometry().getDetectorCount();
	}

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Set the relaxation factor.
void CArtAlgorithm::setLambda(float32 _fLambda)
{
	m_fLambda = _fLambda;
}

//----------------------------------------------------------------------------------------
// Set the order in which the rays will be selected
void CArtAlgorithm::setRayOrder(int* _piProjectionOrder, int* _piDetectorOrder, int _iRayCount)
{
	m_iCurrentRay = 0;
	m_piProjectionOrder.resize(_iRayCount);
	m_piDetectorOrder.resize(_iRayCount);
	for (int i = 0; i < _iRayCount; i++) {
		m_piProjectionOrder[i] = _piProjectionOrder[i];
		m_piDetectorOrder[i] = _piDetectorOrder[i];
	}
}

//----------------------------------------------------------------------------------------
// Iterate
bool CArtAlgorithm::run(int _iNrIterations)
{
	// check initialized
	assert(m_bIsInitialized);
	
	// variables
	int iIteration, iPixel;
	int iUsedPixels, iProjection, iDetector;
	float32 fRayForwardProj, fSumSquaredWeights;
	float32 fProjectionDifference, fBackProjectionFactor;

	// create a pixel buffer
	int iPixelBufferSize = m_pProjector->getProjectionWeightsCount(0);
	SPixelWeight* pPixels = new SPixelWeight[iPixelBufferSize];

	// start iterations
	for (iIteration = _iNrIterations-1; iIteration >= 0; --iIteration) {

		// step0: compute single weight rays
		iProjection = m_piProjectionOrder[m_iCurrentRay];
		iDetector = m_piDetectorOrder[m_iCurrentRay];
		m_iCurrentRay = (m_iCurrentRay + 1) % m_piProjectionOrder.size();

		if (m_bUseSinogramMask && m_pSinogramMask->getData2D()[iProjection][iDetector] == 0) continue;	

		m_pProjector->computeSingleRayWeights(iProjection, iDetector, pPixels, iPixelBufferSize, iUsedPixels);

		// step1: forward projections
		fRayForwardProj = 0.0f;
		fSumSquaredWeights = 0.0f;
		for (iPixel = iUsedPixels-1; iPixel >= 0; --iPixel) {
			if (m_bUseReconstructionMask && m_pReconstructionMask->getDataConst()[pPixels[iPixel].m_iIndex] == 0) continue;

			fRayForwardProj += pPixels[iPixel].m_fWeight * m_pReconstruction->getDataConst()[pPixels[iPixel].m_iIndex];
			fSumSquaredWeights += pPixels[iPixel].m_fWeight * pPixels[iPixel].m_fWeight;
		}
		if (fSumSquaredWeights == 0) continue;

		// step2: difference
		fProjectionDifference = m_pSinogram->getData2D()[iProjection][iDetector] - fRayForwardProj;

		// step3: back projection
		fBackProjectionFactor = m_fLambda * fProjectionDifference / fSumSquaredWeights;
		for (iPixel = iUsedPixels-1; iPixel >= 0; --iPixel) {
			
			// pixel must be loose
			if (m_bUseReconstructionMask && m_pReconstructionMask->getDataConst()[pPixels[iPixel].m_iIndex] == 0) continue;

			// update
			m_pReconstruction->getData()[pPixels[iPixel].m_iIndex] += fBackProjectionFactor * pPixels[iPixel].m_fWeight;
			
			// constraints
			if (m_bUseMinConstraint && m_pReconstruction->getData()[pPixels[iPixel].m_iIndex] < m_fMinValue) {
				m_pReconstruction->getData()[pPixels[iPixel].m_iIndex] = m_fMinValue;
			}
			if (m_bUseMaxConstraint && m_pReconstruction->getData()[pPixels[iPixel].m_iIndex] > m_fMaxValue) {
				m_pReconstruction->getData()[pPixels[iPixel].m_iIndex] = m_fMaxValue;
			}
		}

	}
	delete[] pPixels;

	// update statistics
	m_pReconstruction->updateStatistics();

	return true;
}


//----------------------------------------------------------------------------------------

} // namespace astra
