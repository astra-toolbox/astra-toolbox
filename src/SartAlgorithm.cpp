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

#include "astra/SartAlgorithm.h"

#include "astra/AstraObjectManager.h"
#include "astra/DataProjectorPolicies.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"


//---------------------------------------------------------------------------------------
// Clear - Constructors
void CSartAlgorithm::_clear()
{
	CReconstructionAlgorithm2D::_clear();
	m_piProjectionOrder.clear();
	m_pTotalRayLength = NULL;
	m_pTotalPixelWeight = NULL;

	m_bIsInitialized = false;
	m_iIterationCount = 0;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CSartAlgorithm::clear()
{
	CReconstructionAlgorithm2D::clear();

	m_piProjectionOrder.clear();

	delete m_pTotalRayLength;
	m_pTotalRayLength = NULL;

	delete m_pTotalPixelWeight;
	m_pTotalPixelWeight = NULL;

	delete m_pDiffSinogram;
	m_pDiffSinogram = NULL;

	m_bIsInitialized = false;
	m_iIterationCount = 0;
}

//----------------------------------------------------------------------------------------
// Constructor
CSartAlgorithm::CSartAlgorithm() 
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Constructor
CSartAlgorithm::CSartAlgorithm(CProjector2D* _pProjector, 
							   CFloat32ProjectionData2D* _pSinogram, 
							   CFloat32VolumeData2D* _pReconstruction) 
{
	_clear();
	initialize(_pProjector, _pSinogram, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Constructor
CSartAlgorithm::CSartAlgorithm(CProjector2D* _pProjector, 
							   CFloat32ProjectionData2D* _pSinogram, 
							   CFloat32VolumeData2D* _pReconstruction,
							   int* _piProjectionOrder, 
							   int _iProjectionCount)
{
	_clear();
	initialize(_pProjector, _pSinogram, _pReconstruction, _piProjectionOrder, _iProjectionCount);
}

//----------------------------------------------------------------------------------------
// Destructor
CSartAlgorithm::~CSartAlgorithm() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CSartAlgorithm::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("SartAlgorithm", this, _cfg);
	
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CReconstructionAlgorithm2D::initialize(_cfg)) {
		return false;
	}

	// projection order
	int projectionCount = m_pProjector->getProjectionGeometry().getProjectionAngleCount();
	std::string projOrder;
	if (!CR.getOptionString("ProjectionOrder", projOrder, "sequential"))
		return false;
	if (projOrder == "sequential") {
		m_piProjectionOrder.resize(projectionCount);
		for (int i = 0; i < projectionCount; i++) {
			m_piProjectionOrder[i] = i;
		}
	} else if (projOrder == "random") {
		m_piProjectionOrder.resize(projectionCount);
		for (int i = 0; i < projectionCount; i++) {
			m_piProjectionOrder[i] = i;
		}
		for (int i = 0; i < projectionCount-1; i++) {
			int k = (rand() % (projectionCount - i));
			int t = m_piProjectionOrder[i];
			m_piProjectionOrder[i] = m_piProjectionOrder[i + k];
			m_piProjectionOrder[i + k] = t;
		}
	} else if (projOrder == "custom") {
		if (!CR.getOptionIntArray("ProjectionOrderList", m_piProjectionOrder))
			return false;
	} else {
		ASTRA_ERROR("Unknown ProjectionOrder");
		return false;
	}

	if (!CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f))
		return false;

	// create data objects
	m_pTotalRayLength = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());
	m_pTotalPixelWeight = new CFloat32VolumeData2D(m_pProjector->getVolumeGeometry());
	m_pDiffSinogram = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CSartAlgorithm::initialize(CProjector2D* _pProjector, 
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
	int iProjectionCount = _pProjector->getProjectionGeometry().getProjectionAngleCount();
	m_piProjectionOrder.resize(iProjectionCount);
	for (int i = 0; i < iProjectionCount; i++) {
		m_piProjectionOrder[i] = i;
	}

	// create data objects
	m_pTotalRayLength = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());
	m_pTotalPixelWeight = new CFloat32VolumeData2D(m_pProjector->getVolumeGeometry());
	m_pDiffSinogram = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CSartAlgorithm::initialize(CProjector2D* _pProjector, 
								CFloat32ProjectionData2D* _pSinogram, 
								CFloat32VolumeData2D* _pReconstruction,
								int* _piProjectionOrder, 
								int _iProjectionCount)
{
	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	// ray order
	m_piProjectionOrder.resize(_iProjectionCount);
	for (int i = 0; i < _iProjectionCount; i++) {
		m_piProjectionOrder[i] = _piProjectionOrder[i];
	}

	// create data objects
	m_pTotalRayLength = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());
	m_pTotalPixelWeight = new CFloat32VolumeData2D(m_pProjector->getVolumeGeometry());
	m_pDiffSinogram = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
bool CSartAlgorithm::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm2D::_check(), "SART", "Error in ReconstructionAlgorithm2D initialization");

	// check projection order all within range
	for (unsigned int i = 0; i < m_piProjectionOrder.size(); ++i) {
		ASTRA_CONFIG_CHECK(0 <= m_piProjectionOrder[i] && m_piProjectionOrder[i] < m_pProjector->getProjectionGeometry().getProjectionAngleCount(), "SART", "Projection Order out of range.");
	}

	return true;
}

//----------------------------------------------------------------------------------------
// Iterate
bool CSartAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	// data projectors
	CDataProjectorInterface* pFirstForwardProjector;
	CDataProjectorInterface* pForwardProjector;
	CDataProjectorInterface* pBackProjector;

	m_pTotalRayLength->setData(0.0f);
	m_pTotalPixelWeight->setData(0.0f);

	// backprojection data projector
	pBackProjector = dispatchDataProjector(
			m_pProjector, 
			SinogramMaskPolicy(m_pSinogramMask),														// sinogram mask
			ReconstructionMaskPolicy(m_pReconstructionMask),											// reconstruction mask
			SIRTBPPolicy(m_pReconstruction, m_pDiffSinogram, m_pTotalPixelWeight, m_pTotalRayLength, m_fLambda),	// SIRT backprojection
			m_bUseSinogramMask, m_bUseReconstructionMask, true // options on/off
		); 

	// first time forward projection data projector,
	// also computes total pixel weight and total ray length
	pFirstForwardProjector = dispatchDataProjector(
			m_pProjector, 
			SinogramMaskPolicy(m_pSinogramMask),														// sinogram mask
			ReconstructionMaskPolicy(m_pReconstructionMask),											// reconstruction mask
			Combine3Policy<DiffFPPolicy, TotalPixelWeightPolicy, TotalRayLengthPolicy>(					// 3 basic operations
				DiffFPPolicy(m_pReconstruction, m_pDiffSinogram, m_pSinogram),								// forward projection with difference calculation
				TotalPixelWeightPolicy(m_pTotalPixelWeight),												// calculate the total pixel weights
				TotalRayLengthPolicy(m_pTotalRayLength)),													// calculate the total ray lengths
			m_bUseSinogramMask, m_bUseReconstructionMask, true											 // options on/off
		);

	// forward projection data projector
	pForwardProjector = dispatchDataProjector(
			m_pProjector,
			SinogramMaskPolicy(m_pSinogramMask),														// sinogram mask
			ReconstructionMaskPolicy(m_pReconstructionMask),											// reconstruction mask
			CombinePolicy<DiffFPPolicy, TotalPixelWeightPolicy>(					// 2 basic operations
				DiffFPPolicy(m_pReconstruction, m_pDiffSinogram, m_pSinogram),								// forward projection with difference calculation
				TotalPixelWeightPolicy(m_pTotalPixelWeight)),												// calculate the total pixel weights
			m_bUseSinogramMask, m_bUseReconstructionMask, true											 // options on/off
		);



	// iteration loop
	for (int iIteration = 0; iIteration < _iNrIterations && !shouldAbort(); ++iIteration) {

		int iProjection = m_piProjectionOrder[m_iIterationCount % m_piProjectionOrder.size()];
	
		// forward projection and difference calculation
		m_pTotalPixelWeight->setData(0.0f);
		if ((size_t) iIteration < m_piProjectionOrder.size())
			pFirstForwardProjector->projectSingleProjection(iProjection);
		else
			pForwardProjector->projectSingleProjection(iProjection);
		// backprojection
		pBackProjector->projectSingleProjection(iProjection);
		// update iteration count
		m_iIterationCount++;

		if (m_bUseMinConstraint)
			m_pReconstruction->clampMin(m_fMinValue);
		if (m_bUseMaxConstraint)
			m_pReconstruction->clampMax(m_fMaxValue);
	}


	ASTRA_DELETE(pFirstForwardProjector);
	ASTRA_DELETE(pForwardProjector);
	ASTRA_DELETE(pBackProjector);

	return true;
}
//----------------------------------------------------------------------------------------

} // namespace astra
