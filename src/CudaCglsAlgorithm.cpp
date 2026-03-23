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

#include "astra/CudaCglsAlgorithm.h"

#include "astra/cuda/2d/mem2d.h"
#include "astra/cuda/2d/arith.h"

#include "astra/Logging.h"

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaCglsAlgorithm::CCudaCglsAlgorithm() 
	: m_bBuffersInitialized(false),
	  D_projData(nullptr),
	  D_volData(nullptr),
	  D_z(nullptr),
	  D_p(nullptr),
	  D_r(nullptr),
	  D_w(nullptr),
	  D_volMaskData(nullptr),
	  m_fGamma(0.0f)
{

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaCglsAlgorithm::~CCudaCglsAlgorithm() 
{
	freeBuffers();
}


//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaCglsAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaCglsAlgorithm", this, _cfg);

	if (CR.hasOption("SinogramMaskId")) {
		ASTRA_CONFIG_CHECK(false, "CGLS_CUDA", "Sinogram mask option is not supported.");
	}

	if (!CCudaReconstructionAlgorithm2D::initialize(_cfg))
		return false;

	if (!allocateBuffers())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaCglsAlgorithm::initialize(CProjector2D* _pProjector,
                                    CFloat32ProjectionData2D* _pSinogram, 
                                    CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	if (!CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction))
		return false;

	if (!allocateBuffers())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------

bool CCudaCglsAlgorithm::allocateBuffers()
{
	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	if ((D_volData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_p = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_z = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if (m_bUseReconstructionMask) {
		if ((D_volMaskData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
			return false;
	}
	if ((D_projData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_r = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_w = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
#if 0
	if (m_bUseSinogramMask) {
		if ((D_projMaskData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
			return false;
	}
#endif

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


void CCudaCglsAlgorithm::freeBuffers()
{
	freeGPUMem(D_volData);
	freeGPUMem(D_p);
	freeGPUMem(D_z);
	freeGPUMem(D_volMaskData);
	freeGPUMem(D_projData);
	freeGPUMem(D_r);
	freeGPUMem(D_w);
#if 0
	freeGPUMem(D_projMaskData);
#endif
}

//---------------------------------------------------------------------------------------

bool CCudaCglsAlgorithm::run(int iterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	m_bBuffersInitialized = true;

	ASTRA_ASSERT(m_pSinogram->isFloat32Memory());
	bool ok = astraCUDA::copyToGPUMemory(m_pSinogram, D_projData);

	if (m_bUseReconstructionMask) {
		ASTRA_ASSERT(m_pReconstructionMask->isFloat32Memory());
		ok &= astraCUDA::copyToGPUMemory(m_pReconstructionMask, D_volMaskData);
	}

	ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
	ok &= astraCUDA::copyToGPUMemory(m_pReconstruction, D_volData);

	if (!ok)
		return false;

	// We reset the CGLS algorithm on every iterate call here
	// TODO: We could consider making this an option
	if (true) {

		// copy sinogram
		astraCUDA::assignGPUMemory(D_r, D_projData);

		// r = sino - A*x
		if (m_bUseReconstructionMask) {
			// Use z as temporary storage here since it is unused
			astraCUDA::assignGPUMemory(D_z, D_volData);
			astraCUDA::processData<astraCUDA::opMul>(D_z, D_volMaskData);
			callFP(D_z, D_r, -1.0f);
		} else {
			callFP(D_volData, D_r, -1.0f);
		}

		// p = A'*r
		astraCUDA::zeroGPUMemory(D_p);
		callBP(D_p, D_r, 1.0f);
		if (m_bUseReconstructionMask)
			astraCUDA::processData<astraCUDA::opMul>(D_p, D_volMaskData);

		m_fGamma = astraCUDA::dotProduct2D(D_p);
	}


	// iteration
	for (int iter = 0; iter < iterations && !shouldAbort(); ++iter) {

		// w = A*p
		astraCUDA::zeroGPUMemory(D_w);
		callFP(D_p, D_w, 1.0f);

		// alpha = gamma / <w,w>
		float ww = astraCUDA::dotProduct2D(D_w);
		float alpha = m_fGamma / ww;

		// x += alpha*p
		astraCUDA::processData<astraCUDA::opAddScaled>(D_volData, D_p, alpha);

		// r -= alpha*w
		astraCUDA::processData<astraCUDA::opAddScaled>(D_r, D_w, -alpha);


		// z = A'*r
		astraCUDA::zeroGPUMemory(D_z);
		callBP(D_z, D_r, 1.0f);
		if (m_bUseReconstructionMask)
			astraCUDA::processData<astraCUDA::opMul>(D_z, D_volMaskData);

		float beta = 1.0f / m_fGamma;
		m_fGamma = astraCUDA::dotProduct2D(D_z);
		beta *= m_fGamma;

		// p = z + beta*p
		astraCUDA::processData<astraCUDA::opScaleAndAdd>(D_p, D_z, beta);

	}

	ok &= astraCUDA::copyFromGPUMemory(m_pReconstruction, D_volData);
	if (!ok)
		return false;

	return true;
}

bool CCudaCglsAlgorithm::getResidualNorm(float32& _fNorm)
{
	// Ensure we've performed at least one iteration
	if (!m_bIsInitialized || !m_bBuffersInitialized)
		return false;

	// We can use w and z as temporary storage here since they're not
	// used outside of iterations.

	// copy sinogram to w
	astraCUDA::assignGPUMemory(D_w, D_projData);

	// do FP, subtracting projection from sinogram
	if (m_bUseReconstructionMask) {
		astraCUDA::assignGPUMemory(D_z, D_volData);
		astraCUDA::processData<astraCUDA::opMul>(D_z, D_volMaskData);
		callFP(D_z, D_w, -1.0f);
	} else {
		callFP(D_volData, D_w, -1.0f);
	}

	// compute norm of D_w

	float s = astraCUDA::dotProduct2D(D_w);

	_fNorm = sqrt(s);

	return true;
}



} // namespace astra

#endif // ASTRA_CUDA
