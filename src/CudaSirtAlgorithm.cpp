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

#include "astra/CudaSirtAlgorithm.h"

#include "astra/AstraObjectManager.h"

#include "astra/cuda/2d/mem2d.h"
#include "astra/cuda/2d/arith.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaSirtAlgorithm::CCudaSirtAlgorithm() 
	: m_bBuffersInitialized(false),
	  m_pMinMask(nullptr),
	  m_pMaxMask(nullptr),
	  D_projData(nullptr),
	  D_volData(nullptr),
	  D_tmpProjData(nullptr),
	  D_tmpVolData(nullptr),
	  D_lineWeight(nullptr),
	  D_pixelWeight(nullptr),
	  D_projMaskData(nullptr),
	  D_volMaskData(nullptr),
	  D_minMaskData(nullptr),
	  D_maxMaskData(nullptr),
	  m_fLambda(1.0f)
{

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSirtAlgorithm::~CCudaSirtAlgorithm() 
{
	freeBuffers();
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSirtAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaSirtAlgorithm", this, _cfg);

	if (!CCudaReconstructionAlgorithm2D::initialize(_cfg))
		return false;

	// min/max masks
	int id = -1;
	if (CR.getOptionID("MinMaskId", id)) {
		m_pMinMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}
	if (CR.getOptionID("MaxMaskId", id)) {
		m_pMaxMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}

	bool ok = true;

	ok &= CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f);

	if (!ok)
		return false;

	if (!allocateBuffers())
		return false;

	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaSirtAlgorithm::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram, 
                                     CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	if (!CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction))
		return false;

	m_fLambda = 1.0f;

	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------

bool CCudaSirtAlgorithm::allocateBuffers()
{
	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	if ((D_volData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_tmpVolData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_pixelWeight = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if (m_bUseReconstructionMask) {
		if ((D_volMaskData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
			return false;
	}
	if (m_pMinMask) {
		if ((D_minMaskData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
			return false;
	}
	if (m_pMaxMask) {
		if ((D_maxMaskData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
			return false;
	}
	if ((D_projData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_tmpProjData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
	if (m_bUseSinogramMask) {
		if ((D_projMaskData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
			return false;
	}
	if ((D_lineWeight = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;

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


void CCudaSirtAlgorithm::freeBuffers()
{
	freeGPUMem(D_volData);
	freeGPUMem(D_pixelWeight);
	freeGPUMem(D_tmpVolData);
	freeGPUMem(D_volMaskData);
	freeGPUMem(D_minMaskData);
	freeGPUMem(D_maxMaskData);
	freeGPUMem(D_projData);
	freeGPUMem(D_tmpProjData);
	freeGPUMem(D_lineWeight);
	freeGPUMem(D_projMaskData);
}

//----------------------------------------------------------------------------------------

bool CCudaSirtAlgorithm::precomputeWeights()
{
	astraCUDA::zeroGPUMemory(D_lineWeight);
	if (m_bUseReconstructionMask) {
		callFP(D_volMaskData, D_lineWeight, 1.0f);
	} else {
		astraCUDA::processData<astraCUDA::opSet>(D_tmpVolData, 1.0f);
		callFP(D_tmpVolData, D_lineWeight, 1.0f);
	}
	astraCUDA::processData<astraCUDA::opInvert>(D_lineWeight);

	if (m_bUseSinogramMask) {
		// scale line weights with sinogram mask to zero out masked sinogram pixels
		astraCUDA::processData<astraCUDA::opMul>(D_lineWeight, D_projMaskData);
	}


	astraCUDA::zeroGPUMemory(D_pixelWeight);
	if (m_bUseSinogramMask) {
		callBP(D_pixelWeight, D_projMaskData, 1.0f);
	} else {
		astraCUDA::processData<astraCUDA::opSet>(D_projData, 1.0f);
		callBP(D_pixelWeight, D_projData, 1.0f);
	}
	astraCUDA::processData<astraCUDA::opInvert>(D_pixelWeight);

	if (m_bUseReconstructionMask) {
		// scale pixel weights with mask to zero out masked pixels
		astraCUDA::processData<astraCUDA::opMul>(D_pixelWeight, D_volMaskData);
	}

	// Also fold the relaxation factor into pixel weights
	astraCUDA::processData<astraCUDA::opMul>(D_pixelWeight, m_fLambda);

	return true;
}

//----------------------------------------------------------------------------------------

#if 0
bool SIRT::doSlabCorrections()
{
	// TODO: Decide what to do with this function.
	// Either update it to the new CUDA code architecture and expose it,
	// or remove it?


	// This function compensates for effectively infinitely large slab-like
	// objects of finite thickness 1 in a parallel beam geometry.

	// Each ray through the object has an intersection of length d/cos(alpha).
	// The length of the ray actually intersecting the reconstruction volume is
	// given by D_lineWeight. By dividing by 1/cos(alpha) and multiplying by the
	// lineweights, we correct for this missing attenuation outside of the
	// reconstruction volume, assuming the object is homogeneous.

	// This effectively scales the output values by assuming the thickness d
	// is 1 unit.


	// This function in its current implementation only works if there are no masks.
	// In this case, init() will also have already called precomputeWeights(),
	// so we can use D_lineWeight.
	if (m_bUseReconstructionMask || m_bUseSinogramMask)
		return false;

	// Parallel-beam only
	if (!geometry.isParallel())
		return false;

	// multiply by line weights
	astraCUDA::processData<astraCUDA::opDiv>(D_projData, D_lineWeight);

	SDimensions subdims = dims;
	subdims.iProjAngles = 1;

	// divide by 1/cos(angle)
	// ...but limit the correction to -80/+80 degrees.
	float bound = cosf(1.3963f);
	float* t = (float*)D_projData;
	for (int i = 0; i < dims.iProjAngles; ++i) {
		float angle, detsize, offset;
		getParParameters(geometry.getParallel()[i], dims.iProjDets, angle, detsize, offset);
		float f = fabs(cosf(angle));
		if (f < bound)
			f = bound;

		astraCUDA::processData<astraCUDA::opMul>(t, f, subdims);
		t += sinoPitch;
	}
	return true;
}
#endif


//----------------------------------------------------------------------------------------

bool CCudaSirtAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	bool ok = true;

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	if (!m_bBuffersInitialized) {
		// We can't precompute lineWeights and pixelWeights when using a mask
		if (!m_bUseReconstructionMask && !m_bUseSinogramMask)
			ok &= precomputeWeights();

		if (!ok) {
			return false;
		}

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
	if (m_bUseSinogramMask) {
		ASTRA_ASSERT(m_pSinogramMask->isFloat32Memory());
		ok &= astraCUDA::copyToGPUMemory(m_pSinogramMask, D_projMaskData);
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

	if (m_bUseReconstructionMask || m_bUseSinogramMask)
		ok &= precomputeWeights();

	if (!ok)
		return false;

	for (int iter = 0; iter < _iNrIterations && !astra::shouldAbort(); ++iter) {
		// TODO: Error checking in this loop

		// copy sinogram to projection data
		astraCUDA::assignGPUMemory(D_tmpProjData, D_projData);

		// do FP, subtracting projection from sinogram
		if (m_bUseReconstructionMask) {
				astraCUDA::assignGPUMemory(D_tmpVolData, D_volData);
				astraCUDA::processData<astraCUDA::opMul>(D_tmpVolData, D_volMaskData);
				callFP(D_tmpVolData, D_tmpProjData, -1.0f);
		} else {
				callFP(D_volData, D_tmpProjData, -1.0f);
		}

		astraCUDA::processData<astraCUDA::opMul>(D_tmpProjData, D_lineWeight);

		astraCUDA::zeroGPUMemory(D_tmpVolData);

		callBP(D_tmpVolData, D_tmpProjData, 1.0f);

		// pixel weights also contain the volume mask and relaxation factor
		astraCUDA::processData<astraCUDA::opAddMul>(D_volData, D_pixelWeight, D_tmpVolData);

		if (m_bUseMinConstraint)
			astraCUDA::processData<astraCUDA::opClampMin>(D_volData, m_fMinValue);
		if (m_bUseMaxConstraint)
			astraCUDA::processData<astraCUDA::opClampMax>(D_volData, m_fMaxValue);
		if (D_minMaskData)
			astraCUDA::processData<astraCUDA::opClampMinMask>(D_volData, D_minMaskData);
		if (D_maxMaskData)
			astraCUDA::processData<astraCUDA::opClampMaxMask>(D_volData, D_maxMaskData);
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

bool CCudaSirtAlgorithm::getResidualNorm(float32& _fNorm)
{
	// Ensure we've performed at least one iteration
	if (!m_bIsInitialized || !m_bBuffersInitialized)
		return false;

	// copy sinogram to projection data
	astraCUDA::assignGPUMemory(D_tmpProjData, D_projData);

	// do FP, subtracting projection from sinogram
	if (m_bUseReconstructionMask) {
		astraCUDA::assignGPUMemory(D_tmpVolData, D_volData);
		astraCUDA::processData<astraCUDA::opMul>(D_tmpVolData, D_volMaskData);
		callFP(D_tmpVolData, D_tmpProjData, -1.0f);
	} else {
		callFP(D_volData, D_tmpProjData, -1.0f);
	}

	// compute norm of D_projData
	float s;
	if (!astraCUDA::dotProduct2D(D_tmpProjData, s))
		return false;

	_fNorm = sqrt(s);

	return true;
}


} // namespace astra

#endif // ASTRA_CUDA
