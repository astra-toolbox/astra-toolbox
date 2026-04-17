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

#ifdef ASTRA_CUDA

#include "astra/CudaSartAlgorithm.h"

#include "astra/cuda/2d/mem2d.h"
#include "astra/cuda/2d/arith.h"
#include "astra/cuda/2d/sart.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaSartAlgorithm::CCudaSartAlgorithm() 
	: m_bBuffersInitialized(),
	  D_projData(nullptr),
	  D_volData(nullptr),
	  D_tmpProjData(nullptr),
	  D_tmpVolData(nullptr),
	  D_volMaskData(nullptr),
	  D_lineWeight(nullptr),
	  m_fLambda(1.0f),
	  m_iIteration(0)
{

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSartAlgorithm::~CCudaSartAlgorithm() 
{
	freeBuffers();
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSartAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaSartAlgorithm", this, _cfg);

	if (!CCudaReconstructionAlgorithm2D::initialize(_cfg))
		return false;

	if (CR.hasOption("SinogramMaskId")) {
		ASTRA_CONFIG_CHECK(false, "SART_CUDA", "Sinogram mask option is not supported.");
	}

	// projection order
	int projectionCount = m_pSinogram->getGeometry().getProjectionAngleCount();
	std::string projOrder;
	if (!CR.getOptionString("ProjectionOrder", projOrder, "random"))
		return false;
	if (projOrder == "sequential") {
		m_projectionOrder.resize(projectionCount);
		for (int i = 0; i < projectionCount; i++) {
			m_projectionOrder[i] = i;
		}
	} else if (projOrder == "random") {
		m_projectionOrder.resize(projectionCount);
		for (int i = 0; i < projectionCount; i++) {
			m_projectionOrder[i] = i;
		}
		for (int i = 0; i < projectionCount-1; i++) {
			int k = (rand() % (projectionCount - i));
			int t = m_projectionOrder[i];
			m_projectionOrder[i] = m_projectionOrder[i + k];
			m_projectionOrder[i + k] = t;
		}
	} else if (projOrder == "custom") {
		// NB: For custom orders, length of vector can be different than projectionCount
		if (!CR.getOptionIntArray("ProjectionOrderList", m_projectionOrder))
			return false;
	} else {
		ASTRA_ERROR("Unknown ProjectionOrder");
		return false;
	}

	if (!CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f))
		return false;

	if (!allocateBuffers())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaSartAlgorithm::initialize(CProjector2D* _pProjector,
                                    CFloat32ProjectionData2D* _pSinogram, 
                                    CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	if (!CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction))
		return false;

	m_fLambda = 1.0f;

	if (!allocateBuffers())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------

bool CCudaSartAlgorithm::allocateBuffers()
{
	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	if ((D_volData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if (m_bUseReconstructionMask) {
		if ((D_volMaskData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
			return false;

		// Only allocate D_tmpVolData if we use a mask
		if ((D_tmpVolData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
			return false;
	}
	if ((D_projData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_lineWeight = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;

	astra::CDataStorage *storage = astraCUDA::allocateGPUMemory(m_pSinogram->getDetectorCount(), 1, astraCUDA::INIT_ZERO);
	if (!storage)
		return false;
	D_tmpProjData = new astra::CData2D(m_pSinogram->getDetectorCount(), 1, storage);

	return true;
}

// TODO: Centralize this somehow
// (By making GPU DataStorage objects keep track of if they should free their storage in their destructor)
static void freeGPUMem(CData2D*& ptr)
{
	if (ptr) {
		astraCUDA::freeGPUMemory(ptr);
		delete ptr;
		ptr = nullptr;
	}
}


void CCudaSartAlgorithm::freeBuffers()
{
	freeGPUMem(D_volData);
	freeGPUMem(D_tmpVolData);
	freeGPUMem(D_volMaskData);
	freeGPUMem(D_projData);
	freeGPUMem(D_tmpProjData);
	freeGPUMem(D_lineWeight);
}

//----------------------------------------------------------------------------------------

bool CCudaSartAlgorithm::precomputeWeights()
{
	astraCUDA::zeroGPUMemory(D_lineWeight);
	if (m_bUseReconstructionMask) {
		callFP(D_volMaskData, D_lineWeight, 1.0f);
	} else {
		// Allocate tmpData temporarily
		CData2D *D_tmpData = astraCUDA::createGPUData2DLike(m_pReconstruction);
		if (!D_tmpData)
			return false;

		astraCUDA::processData<astraCUDA::opSet>(D_tmpData, 1.0f);
		callFP(D_tmpData, D_lineWeight, 1.0f);

		freeGPUMem(D_tmpData);
	}
	astraCUDA::processData<astraCUDA::opInvert>(D_lineWeight);

	return true;
}


//----------------------------------------------------------------------------------------

bool CCudaSartAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	bool ok = true;

	if (!m_bBuffersInitialized) {
		if (!m_bUseReconstructionMask)
			ok = precomputeWeights();
		if (!ok)
			return false;
		m_bBuffersInitialized = true;
	}

	if (m_pSinogram->isFloat32Memory()) {
		ok &= astraCUDA::copyToGPUMemory(m_pSinogram, D_projData);
	} else if (m_pSinogram->isFloat32GPU()) {
		// TODO: re-use memory instead of copying
		// (need to ensure everything works when pitches are not consistent)
		ok &= astraCUDA::assignGPUMemory(D_projData, m_pSinogram);
	} else {
		ok = false;
	}

	if (!ok)
		return false;

	if (m_bUseReconstructionMask) {
		ASTRA_ASSERT(m_pReconstructionMask->isFloat32Memory());
		ok &= astraCUDA::copyToGPUMemory(m_pReconstructionMask, D_volMaskData);
	}

	if (m_pReconstruction->isFloat32Memory()) {
		ok &= astraCUDA::copyToGPUMemory(m_pReconstruction, D_volData);
	} else if (m_pReconstruction->isFloat32GPU()) {
		// TODO: re-use memory instead of copying
		// (need to ensure everything works when pitches are not consistent)
		ok &= astraCUDA::assignGPUMemory(D_volData, m_pReconstruction);
	} else {
		ok = false;
	}

	if (!ok)
		return false;

	if (m_bUseReconstructionMask) {
		ok &= precomputeWeights();
		if (!ok)
			return false;
	}

	// iteration
	for (int iter = 0; iter < _iNrIterations && !shouldAbort(); ++iter) {

		int angle;
		if (!m_projectionOrder.empty()) {
			angle = m_projectionOrder[m_iIteration % m_projectionOrder.size()];
		} else {
			angle = m_iIteration % m_pSinogram->getAngleCount();
		}

		// copy one line of sinogram to projection data
		astraCUDA::copy_SART(D_tmpProjData, D_projData, angle);

		// do FP, subtracting projection from sinogram
		if (m_bUseReconstructionMask) {
			astraCUDA::assignGPUMemory(D_tmpVolData, D_volData);
			astraCUDA::processData<astraCUDA::opMul>(D_tmpVolData, D_volMaskData);
			FP_SART(D_tmpVolData, D_tmpProjData, angle, -1.0f);
		} else {
			FP_SART(D_volData, D_tmpProjData, angle, -1.0f);
		}

		astraCUDA::mul_SART(D_tmpProjData, D_lineWeight, angle);
		if (m_bUseReconstructionMask) {
			// BP, mask, and add back
			// TODO: Try putting the masking directly in the BP
			astraCUDA::zeroGPUMemory(D_tmpVolData);
			BP_SART(D_tmpVolData, D_tmpProjData, angle, m_fLambda);
			astraCUDA::processData<astraCUDA::opAddMul>(D_volData, D_volMaskData, D_tmpVolData);
		} else {
			BP_SART(D_volData, D_tmpProjData, angle, m_fLambda);
		}

		if (m_bUseMinConstraint)
			astraCUDA::processData<astraCUDA::opClampMin>(D_volData, m_fMinValue);
		if (m_bUseMaxConstraint)
			astraCUDA::processData<astraCUDA::opClampMax>(D_volData, m_fMaxValue);

		m_iIteration++;

	}

	if (ok) {
		if (m_pReconstruction->isFloat32Memory()) {
			ok &= astraCUDA::copyFromGPUMemory(m_pReconstruction, D_volData);
		} else if (m_pReconstruction->isFloat32GPU()) {
			// TODO: re-use memory instead of copying
			// (need to ensure everything works when pitches are not consistent)
			ok &= astraCUDA::assignGPUMemory(m_pReconstruction, D_volData);
		} else {
			ok = false;
		}
	}

	if (!ok)
		return false;

	return true;
}

//----------------------------------------------------------------------------------------

bool CCudaSartAlgorithm::FP_SART(const astra::CData2D *D_vol, astra::CData2D *D_proj, int angle, float fScale)
{
	astraCUDA::SProjectorParams2D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA::FP_SART(D_proj, D_vol, m_geometry, p, angle);
}

bool CCudaSartAlgorithm::BP_SART(astra::CData2D *D_vol, const astra::CData2D *D_proj, int angle, float fScale)
{
	astraCUDA::SProjectorParams2D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA::BP_SART(D_proj, D_vol, m_geometry, p, angle);
}


//----------------------------------------------------------------------------------------

bool CCudaSartAlgorithm::getResidualNorm(float32& _fNorm)
{
	// Ensure we've performed at least one iteration
	if (!m_bIsInitialized || !m_bBuffersInitialized)
		return false;

	CData2D *D_p;
	if ((D_p = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;

	// copy sinogram to D_p
	astraCUDA::assignGPUMemory(D_p, D_projData);

	// do FP, subtracting projection from sinogram
	if (m_bUseReconstructionMask) {
			astraCUDA::assignGPUMemory(D_tmpVolData, D_volData);
			astraCUDA::processData<astraCUDA::opMul>(D_tmpVolData, D_volMaskData);
			callFP(D_tmpVolData, D_p, -1.0f);
	} else {
			callFP(D_volData, D_p, -1.0f);
	}

	// compute norm of D_p
	float s;
	if (!astraCUDA::dotProduct2D(D_p, s)) {
		freeGPUMem(D_p);
		return false;
	}

	freeGPUMem(D_p);

	_fNorm = sqrt(s);

	return true;
}

} // namespace astra

#endif // ASTRA_CUDA
