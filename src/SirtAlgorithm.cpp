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

#include "astra/SirtAlgorithm.h"

#include "astra/AstraObjectManager.h"
#include "astra/DataProjectorPolicies.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CSirtAlgorithm::type = "SIRT";

//----------------------------------------------------------------------------------------
// Constructor
CSirtAlgorithm::CSirtAlgorithm() 
{
	_clear();
}

//---------------------------------------------------------------------------------------
// Initialize - C++
CSirtAlgorithm::CSirtAlgorithm(CProjector2D* _pProjector, 
							   CFloat32ProjectionData2D* _pSinogram, 
							   CFloat32VolumeData2D* _pReconstruction)
{
	_clear();
	initialize(_pProjector, _pSinogram, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CSirtAlgorithm::~CSirtAlgorithm() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CSirtAlgorithm::_clear()
{
	CReconstructionAlgorithm2D::_clear();
	m_bIsInitialized = false;

	m_pTotalRayLength = NULL;
	m_pTotalPixelWeight = NULL;
	m_pDiffSinogram = NULL;
	m_pTmpVolume = NULL;

	m_fLambda = 1.0f;
	m_iIterationCount = 0;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CSirtAlgorithm::clear()
{
	CReconstructionAlgorithm2D::_clear();
	m_bIsInitialized = false;

	ASTRA_DELETE(m_pTotalRayLength);
	ASTRA_DELETE(m_pTotalPixelWeight);
	ASTRA_DELETE(m_pDiffSinogram);
	ASTRA_DELETE(m_pTmpVolume);

	m_fLambda = 1.0f;
	m_iIterationCount = 0;
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
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("SirtAlgorithm", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CReconstructionAlgorithm2D::initialize(_cfg)) {
		return false;
	}

	m_fLambda = _cfg.self.getOptionNumerical("Relaxation", 1.0f);
	CC.markOptionParsed("Relaxation");

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
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

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
	m_pTotalRayLength = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());
	m_pTotalPixelWeight = new CFloat32VolumeData2D(m_pProjector->getVolumeGeometry());
	m_pDiffSinogram = new CFloat32ProjectionData2D(m_pProjector->getProjectionGeometry());
	m_pTmpVolume = new CFloat32VolumeData2D(m_pProjector->getVolumeGeometry());
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CSirtAlgorithm::getInformation() 
{
	map<string, boost::any> res;
	return mergeMap<string,boost::any>(CReconstructionAlgorithm2D::getInformation(), res);
};

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CSirtAlgorithm::getInformation(std::string _sIdentifier) 
{
	return CAlgorithm::getInformation(_sIdentifier);
};

//----------------------------------------------------------------------------------------
// Iterate
void CSirtAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	m_bShouldAbort = false;

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

	float32* pfT = m_pTotalPixelWeight->getData();
	for (int i = 0; i < m_pTotalPixelWeight->getSize(); ++i) {
		float32 x = pfT[i];
		if (x < -eps || x > eps)
			x = 1.0f / x;
		else
			x = 0.0f;
		pfT[i] = m_fLambda * x;
	}
	pfT = m_pTotalRayLength->getData();
	for (int i = 0; i < m_pTotalRayLength->getSize(); ++i) {
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
	for (; iIteration < _iNrIterations && !m_bShouldAbort; ++iIteration) {
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
}
//----------------------------------------------------------------------------------------

} // namespace astra
