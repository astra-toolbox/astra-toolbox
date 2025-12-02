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

#include "astra/SirtAlgorithm.h"

#include "astra/AstraObjectManager.h"
#include "astra/DataProjectorPolicies.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

//----------------------------------------------------------------------------------------
// Constructor
CSirtAlgorithm::CSirtAlgorithm()
	: m_pTotalRayLength(nullptr),
	  m_pTotalPixelWeight(nullptr),
	  m_pDiffSinogram(nullptr),
	  m_pTmpVolume(nullptr),
	  m_iIterationCount(0),
	  m_fLambda(1.0f)
{

}

//---------------------------------------------------------------------------------------
// Initialize - C++
CSirtAlgorithm::CSirtAlgorithm(CProjector2D* _pProjector, 
                               CFloat32ProjectionData2D* _pSinogram, 
                               CFloat32VolumeData2D* _pReconstruction)
	: CSirtAlgorithm()
{
	initialize(_pProjector, _pSinogram, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CSirtAlgorithm::~CSirtAlgorithm() 
{
	delete m_pTotalRayLength;
	delete m_pTotalPixelWeight;
	delete m_pDiffSinogram;
	delete m_pTmpVolume;
}

//----------------------------------------------------------------------------------------
// Check
bool CSirtAlgorithm::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm2D::_check(), "SIRT", "Error in ReconstructionAlgorithm2D initialization");

	ASTRA_CONFIG_CHECK(m_pTotalRayLength, "SIRT", "Invalid TotalRayLength Object");
	ASTRA_CONFIG_CHECK(m_pTotalRayLength->isInitialized(), "SIRT", "Invalid TotalRayLength Object");
	ASTRA_CONFIG_CHECK(m_pTotalPixelWeight, "SIRT", "Invalid TotalPixelWeight Object");
	ASTRA_CONFIG_CHECK(m_pTotalPixelWeight->isInitialized(), "SIRT", "Invalid TotalPixelWeight Object");
	ASTRA_CONFIG_CHECK(m_pDiffSinogram, "SIRT", "Invalid DiffSinogram Object");
	ASTRA_CONFIG_CHECK(m_pDiffSinogram->isInitialized(), "SIRT", "Invalid DiffSinogram Object");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CSirtAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("SirtAlgorithm", this, _cfg);

	// initialization of parent class
	if (!CReconstructionAlgorithm2D::initialize(_cfg)) {
		return false;
	}

	bool ok = true;

	ok &= CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f);

	if (!ok)
		return false;

	// init data objects and data projectors
	_init();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CSirtAlgorithm::initialize(CProjector2D* _pProjector, 
								CFloat32ProjectionData2D* _pSinogram, 
								CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	m_fLambda = 1.0f;

	// init data objects and data projectors
	_init();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize Data Projectors - private
void CSirtAlgorithm::_init()
{
	// create data objects
	m_pTotalRayLength = createCFloat32ProjectionData2DMemory(m_pProjector->getProjectionGeometry());
	m_pTotalPixelWeight = createCFloat32VolumeData2DMemory(m_pProjector->getVolumeGeometry());
	m_pDiffSinogram = createCFloat32ProjectionData2DMemory(m_pProjector->getProjectionGeometry());
	m_pTmpVolume = createCFloat32VolumeData2DMemory(m_pProjector->getVolumeGeometry());
}

//----------------------------------------------------------------------------------------
// Iterate
bool CSirtAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	int iIteration = 0;

	// data projectors
	CDataProjectorInterface* pForwardProjector;
	CDataProjectorInterface* pBackProjector;
	CDataProjectorInterface* pFirstForwardProjector;

	m_pTotalRayLength->setData(0.0f);
	m_pTotalPixelWeight->setData(0.0f);

	// forward projection data projector
	pForwardProjector = dispatchDataProjector(
		m_pProjector, 
			SinogramMaskPolicy(m_pSinogramMask),														// sinogram mask
			ReconstructionMaskPolicy(m_pReconstructionMask),											// reconstruction mask
			DiffFPPolicy(m_pReconstruction, m_pDiffSinogram, m_pSinogram),								// forward projection with difference calculation
			m_bUseSinogramMask, m_bUseReconstructionMask, true											// options on/off
		); 

	// backprojection data projector
	pBackProjector = dispatchDataProjector(
			m_pProjector, 
			SinogramMaskPolicy(m_pSinogramMask),														// sinogram mask
			ReconstructionMaskPolicy(m_pReconstructionMask),											// reconstruction mask
			DefaultBPPolicy(m_pTmpVolume, m_pDiffSinogram), // backprojection
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



	// forward projection, difference calculation and raylength/pixelweight computation
	pFirstForwardProjector->project();

	float32* pfT = m_pTotalPixelWeight->getFloat32Memory();
	for (size_t i = 0; i < m_pTotalPixelWeight->getSize(); ++i) {
		float32 x = pfT[i];
		if (x < -eps || x > eps)
			x = 1.0f / x;
		else
			x = 0.0f;
		pfT[i] = m_fLambda * x;
	}
	pfT = m_pTotalRayLength->getFloat32Memory();
	for (size_t i = 0; i < m_pTotalRayLength->getSize(); ++i) {
		float32 x = pfT[i];
		if (x < -eps || x > eps)
			x = 1.0f / x;
		else
			x = 0.0f;
		pfT[i] = x;
	}

	// divide by line weights
	(*m_pDiffSinogram) *= (*m_pTotalRayLength);

	// backprojection
	m_pTmpVolume->setData(0.0f);
	pBackProjector->project();

	// divide by pixel weights
	(*m_pTmpVolume) *= (*m_pTotalPixelWeight);
	(*m_pReconstruction) += (*m_pTmpVolume);

	if (m_bUseMinConstraint)
		m_pReconstruction->clampMin(m_fMinValue);
	if (m_bUseMaxConstraint)
		m_pReconstruction->clampMax(m_fMaxValue);

	// update iteration count
	m_iIterationCount++;
	iIteration++;


	

	// iteration loop
	for (; iIteration < _iNrIterations && !shouldAbort(); ++iIteration) {
		// forward projection and difference calculation
		pForwardProjector->project();

		// divide by line weights
		(*m_pDiffSinogram) *= (*m_pTotalRayLength);


		// backprojection
		m_pTmpVolume->setData(0.0f);
		pBackProjector->project();

		// multiply with relaxation factor divided by pixel weights
		(*m_pTmpVolume) *= (*m_pTotalPixelWeight);
		(*m_pReconstruction) += (*m_pTmpVolume);

		if (m_bUseMinConstraint)
			m_pReconstruction->clampMin(m_fMinValue);
		if (m_bUseMaxConstraint)
			m_pReconstruction->clampMax(m_fMaxValue);

		// update iteration count
		m_iIterationCount++;
	}


	ASTRA_DELETE(pForwardProjector);
	ASTRA_DELETE(pBackProjector);
	ASTRA_DELETE(pFirstForwardProjector);

	return true;
}
//----------------------------------------------------------------------------------------

} // namespace astra
