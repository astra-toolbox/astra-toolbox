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

#include "astra/CudaEMAlgorithm.h"

#include "astra/cuda/2d/mem2d.h"
#include "astra/cuda/2d/arith.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaEMAlgorithm::CCudaEMAlgorithm() 
	: m_bBuffersInitialized(false),
	  D_projData(nullptr),
	  D_volData(nullptr),
	  D_pixelWeight(nullptr),
	  D_tmpProjData(nullptr),
	  D_tmpVolData(nullptr)
{

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaEMAlgorithm::~CCudaEMAlgorithm() 
{
	freeBuffers();
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaEMAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaEMAlgorithm", this, _cfg);

	if (CR.hasOption("SinogramMaskId")) {
		ASTRA_CONFIG_CHECK(false, "EM_CUDA", "Sinogram mask option is not supported.");
	}
	if (CR.hasOption("ReconstructionMaskId")) {
		ASTRA_CONFIG_CHECK(false, "EM_CUDA", "Reconstruction mask option is not supported.");
	}

	if (!CCudaReconstructionAlgorithm2D::initialize(_cfg))
		return false;

	if (!allocateBuffers())
		return false;

	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaEMAlgorithm::initialize(CProjector2D* _pProjector,
                                  CFloat32ProjectionData2D* _pSinogram, 
                                  CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	if (!CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction))
		return false;

	if (!allocateBuffers())
		return false;

	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------

bool CCudaEMAlgorithm::allocateBuffers()
{
	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	if ((D_volData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_tmpVolData = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_pixelWeight = astraCUDA::createGPUData2DLike(m_pReconstruction)) == nullptr)
		return false;

	if ((D_projData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_tmpProjData = astraCUDA::createGPUData2DLike(m_pSinogram)) == nullptr)
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


void CCudaEMAlgorithm::freeBuffers()
{
	freeGPUMem(D_volData);
	freeGPUMem(D_tmpVolData);
	freeGPUMem(D_pixelWeight);
	freeGPUMem(D_projData);
	freeGPUMem(D_tmpProjData);
}

//----------------------------------------------------------------------------------------

bool CCudaEMAlgorithm::precomputeWeights()
{
	astraCUDA::zeroGPUMemory(D_pixelWeight);
#if 0
	if (useSinogramMask) {
		callBP(D_pixelWeight, pixelPitch, D_smaskData, smaskPitch);
	} else
#endif
	{
		astraCUDA::processData<astraCUDA::opSet>(D_tmpProjData, 1.0f);
		callBP(D_pixelWeight, D_tmpProjData, 1.0f);
	}
	astraCUDA::processData<astraCUDA::opInvert>(D_pixelWeight);

#if 0
	if (useVolumeMask) {
		// scale pixel weights with mask to zero out masked pixels
		processVol<opMul>(D_pixelWeight, D_maskData, pixelPitch, dims);
	}
#endif

	return true;
}

//----------------------------------------------------------------------------------------

bool CCudaEMAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	if (!m_bBuffersInitialized) {
		precomputeWeights();
		m_bBuffersInitialized = true;
	}

	ASTRA_ASSERT(m_pSinogram->isFloat32Memory());
	bool ok = astraCUDA::copyToGPUMemory(m_pSinogram, D_projData);

	ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
	ok &= astraCUDA::copyToGPUMemory(m_pReconstruction, D_volData);

	if (!ok)
		return false;

	// iteration
	for (int iter = 0; iter < _iNrIterations && !shouldAbort(); ++iter) {

		// Do FP of volData  (into tmpProjData)
		astraCUDA::zeroGPUMemory(D_tmpProjData);
		callFP(D_volData, D_tmpProjData, 1.0f);

		// Divide sinogram by FP (into tmpProjData)
		astraCUDA::processData<astraCUDA::opDividedBy>(D_tmpProjData, D_projData);

		// Do BP of tmpProjData into tmpVolData
		astraCUDA::zeroGPUMemory(D_tmpVolData);
		callBP(D_tmpVolData, D_tmpProjData, 1.0f);

		// Multiply volumeData with tmpData divided by pixel weights
		astraCUDA::processData<astraCUDA::opMul2>(D_volData, D_tmpVolData, D_pixelWeight);

	}

	ok &= astraCUDA::copyFromGPUMemory(m_pReconstruction, D_volData);
	if (!ok)
		return false;

	return true;

}

bool CCudaEMAlgorithm::getResidualNorm(float32& _fNorm)
{
	// Ensure we've performed at least one iteration
	if (!m_bIsInitialized || !m_bBuffersInitialized)
		return false;

	// copy sinogram to projection data
	astraCUDA::assignGPUMemory(D_tmpProjData, D_projData);
	callFP(D_volData, D_tmpProjData, -1.0f);

	// compute norm of D_tmpProjData

	float s = astraCUDA::dotProduct2D(D_tmpProjData);

	_fNorm = sqrt(s);

	return true;
}



} // namespace astra

#endif // ASTRA_CUDA
