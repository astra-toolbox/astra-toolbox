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

#include "astra/CudaCglsAlgorithm3D.h"

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/VolumeGeometry3D.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/astra3d.h"
#include "astra/cuda/3d/mem3d.h"
#include "astra/cuda/3d/arith3d.h"


using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaCglsAlgorithm3D::CCudaCglsAlgorithm3D() 
	: m_bBuffersInitialized(false),
	  m_iGPUIndex(-1),
	  D_projData(nullptr),
	  D_volData(nullptr),
	  D_w(nullptr),
	  D_z(nullptr),
	  D_r(nullptr),
	  D_p(nullptr),
	  D_volMaskData(nullptr)
{

}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaCglsAlgorithm3D::CCudaCglsAlgorithm3D(CProjector3D* _pProjector,
                                           CFloat32ProjectionData3D* _pProjectionData,
                                           CFloat32VolumeData3D* _pReconstruction)
	: CCudaCglsAlgorithm3D()
{
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaCglsAlgorithm3D::~CCudaCglsAlgorithm3D() 
{
	freeBuffers();
}


//---------------------------------------------------------------------------------------
// Check
bool CCudaCglsAlgorithm3D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm3D::_check(), "CGLS3D", "Error in ReconstructionAlgorithm3D initialization");

	ASTRA_CONFIG_CHECK(!m_bUseMinConstraint, "CGLS3D", "MinConstraint is not supported");
	ASTRA_CONFIG_CHECK(!m_bUseMaxConstraint, "CGLS3D", "MaxConstraint is not supported");
	ASTRA_CONFIG_CHECK(!m_bUseSinogramMask, "CGLS3D", "SinogramMask is not supported");

	ASTRA_CONFIG_CHECK(m_pSinogram->isFloat32Memory(), "CGLS3D", "Projection data object not a float32 host memory object");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isFloat32Memory(), "CGLS3D", "Reconstruction data object not a float32 host memory object");

	ASTRA_CONFIG_CHECK(!m_bUseReconstructionMask || m_pReconstructionMask->isFloat32Memory(), "CGLS3D", "Reconstruction mask object not a float32 host memory object");

	return true;
}

//---------------------------------------------------------------------------------------
void CCudaCglsAlgorithm3D::initializeFromProjector()
{
	m_params.iRaysPerVoxelDim = 1;
	m_params.iRaysPerDetDim = 1;
	m_params.projKernel = astraCUDA3d::ker3d_default;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to CGLS3D_CUDA");
		}
	} else {
		// TODO: Warn if multiple GPUs specified but only one is used?
		std::vector<int> indices = pCudaProjector->getGPUIndices();
		if (!indices.empty())
			m_iGPUIndex = indices[0];

		m_params.iRaysPerVoxelDim = pCudaProjector->getVoxelSuperSampling();
		m_params.iRaysPerDetDim = pCudaProjector->getDetectorSuperSampling();
		m_params.projKernel = pCudaProjector->getProjectionKernel();
	}
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaCglsAlgorithm3D::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaCglsAlgorithm3D", this, _cfg);

	// initialization of parent class
	if (!CReconstructionAlgorithm3D::initialize(_cfg)) {
		return false;
	}

	initializeFromProjector();

	bool ok = true;

	// Deprecated options
	ok &= CR.getOptionUInt("VoxelSuperSampling", m_params.iRaysPerVoxelDim, m_params.iRaysPerVoxelDim);
	ok &= CR.getOptionUInt("DetectorSuperSampling", m_params.iRaysPerDetDim, m_params.iRaysPerDetDim);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, m_iGPUIndex);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, m_iGPUIndex);
	if (!ok)
		return false;

	if (m_pSinogram->getGeometry().isOfType("cyl_cone_vec")
	    && (m_params.iRaysPerDetDim > 1 || m_params.iRaysPerVoxelDim > 1)) {
		ASTRA_CONFIG_CHECK(false, "CGLS3D_CUDA",
						   "Detector/voxel supersampling is not supported for cyl_cone_vec geometry.");
	}

	if (!allocateBuffers())
		return false;
	if (!setupGeometry())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaCglsAlgorithm3D::initialize(CProjector3D* _pProjector,
                                      CFloat32ProjectionData3D* _pSinogram,
                                      CFloat32VolumeData3D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	initializeFromProjector();

	if (!allocateBuffers())
		return false;
	if (!setupGeometry())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
bool CCudaCglsAlgorithm3D::setupGeometry()
{
	m_geometry = astra::convertAstraGeometry(&m_pReconstruction->getGeometry(), &m_pSinogram->getGeometry());
	m_params.volScale = m_geometry.getVolScale();

	return m_geometry.isValid();
}

bool CCudaCglsAlgorithm3D::allocateBuffers()
{
	if (m_iGPUIndex != -1)
		astraCUDA3d::setGPUIndex(m_iGPUIndex);

	if ((D_volData = astraCUDA3d::createGPUData3DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_p = astraCUDA3d::createGPUData3DLike(m_pReconstruction)) == nullptr)
		return false;
	if ((D_z = astraCUDA3d::createGPUData3DLike(m_pReconstruction)) == nullptr)
		return false;
	if (m_bUseReconstructionMask) {
		if ((D_volMaskData = astraCUDA3d::createGPUData3DLike(m_pReconstruction)) == nullptr)
			return false;
	}
	if ((D_projData = astraCUDA3d::createGPUData3DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_r = astraCUDA3d::createGPUData3DLike(m_pSinogram)) == nullptr)
		return false;
	if ((D_w = astraCUDA3d::createGPUData3DLike(m_pSinogram)) == nullptr)
		return false;
#if 0
	if (m_bUseSinogramMask) {
		if ((D_projMaskData = astraCUDA3d::createGPUData3DLike(m_pSinogram)) == nullptr)
			return false;
	}
#endif

	return true;
}

// TODO: Centralize this somehow
// (By making GPU DataStorage objects keep track of if they should free their storage in their destructor)
static void freeGPUMem(CData3D*& ptr)
{
	if (ptr) {
		astraCUDA3d::freeGPUMemory(ptr);
		delete ptr;
		ptr = nullptr;
	}
}

void CCudaCglsAlgorithm3D::freeBuffers()
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

//----------------------------------------------------------------------------------------
// Iterate
bool CCudaCglsAlgorithm3D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	bool ok = true;

	if (m_iGPUIndex != -1)
		astraCUDA3d::setGPUIndex(m_iGPUIndex);

	m_bBuffersInitialized = true;

	ASTRA_ASSERT(m_pSinogram->isFloat32Memory());

	ok &= astraCUDA3d::copyToGPUMemory(m_pSinogram, D_projData);

	ASTRA_ASSERT(ok);

	if (m_bUseReconstructionMask) {
		ASTRA_ASSERT(m_pReconstructionMask->isFloat32Memory());
		ok &= astraCUDA3d::copyToGPUMemory(m_pReconstructionMask, D_volMaskData);
	}


	ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
	ok &= astraCUDA3d::copyToGPUMemory(m_pReconstruction, D_volData);

	ASTRA_ASSERT(ok);

	float gamma = 0.0f;

	// We reset the CGLS algorithm on every iterate call here
	// TODO: We could consider making this an option
	if (true) {

		// r = sino - A*x
		astraCUDA3d::assignGPUMemory(D_r, D_projData);
		if (m_bUseReconstructionMask) {
			astraCUDA3d::assignGPUMemory(D_z, D_volData);
			astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_z, D_volMaskData);
			callFP(D_z, D_r, -1.0f);
		} else {
			callFP(D_volData, D_r, -1.0f);
		}

		// p = A'*r
		astraCUDA3d::zeroGPUMemory(D_p);
		callBP(D_p, D_r, 1.0f);
		if (m_bUseReconstructionMask)
			astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_p, D_volMaskData);

		astraCUDA3d::dotProduct3D(D_p, gamma);
	}



	for (int iter = 0; iter < _iNrIterations && !astra::shouldAbort(); ++iter) {

		// w = A*p
		astraCUDA3d::zeroGPUMemory(D_w);
		callFP(D_p, D_w, 1.0f);

		// alpha = gamma / <w,w>
		float ww;
		astraCUDA3d::dotProduct3D(D_w, ww);
		float alpha = gamma / ww;

		// x += alpha*p
		astraCUDA3d::processVol3D<astraCUDA3d::opAddScaled>(D_volData, D_p, alpha);

		// r -= alpha*w
		astraCUDA3d::processVol3D<astraCUDA3d::opAddScaled>(D_r, D_w, -alpha);

		// z = A'*r
		astraCUDA3d::zeroGPUMemory(D_z);
		callBP(D_z, D_r, 1.0f);
		if (m_bUseReconstructionMask)
			astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_z, D_volMaskData);

		float beta = 1.0f / gamma;
		astraCUDA3d::dotProduct3D(D_z, gamma);

		beta *= gamma;

		// p = z + beta*p
		astraCUDA3d::processVol3D<astraCUDA3d::opScaleAndAdd>(D_p, D_z, beta);
	}

	ok &= astraCUDA3d::copyFromGPUMemory(m_pReconstruction, D_volData);

	return ok;
}
//----------------------------------------------------------------------------------------
bool CCudaCglsAlgorithm3D::getResidualNorm(float32& _fNorm)
{
	if (!m_bIsInitialized || !m_bBuffersInitialized)
		return false;

	if (m_iGPUIndex != -1)
		astraCUDA3d::setGPUIndex(m_iGPUIndex);

	bool ok = true;

	// We can use w and z as temporary buffers, since they are not used
	// outside of iterations.

	// copy projection data to w
	ok &= astraCUDA3d::assignGPUMemory(D_w, D_projData);

	if (!ok)
		return false;

	// do FP, subtracting projection from sinogram
	if (m_bUseReconstructionMask) {
		ok &= astraCUDA3d::assignGPUMemory(D_z, D_volData);
		ok &= astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_z, D_volMaskData);
		ok &= callFP(D_z, D_w, -1.0f);
	} else {
		ok &= callFP(D_volData, D_w, -1.0f);
	}

	if (!ok)
		return false;

	float s;
	if (!astraCUDA3d::dotProduct3D(D_w, s))
		return false;

	_fNorm = sqrt(s);

	return true;
}
//----------------------------------------------------------------------------------------
bool CCudaCglsAlgorithm3D::callFP(const CData3D *D_vol, CData3D *D_proj, float fScale)
{
	astraCUDA3d::SProjectorParams3D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA3d::FP(D_proj, D_vol, m_geometry, p);
}

bool CCudaCglsAlgorithm3D::callBP(CData3D *D_vol, const CData3D *D_proj, float fScale)
{
	astraCUDA3d::SProjectorParams3D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA3d::BP(D_proj, D_vol, m_geometry, p);
}




} // namespace astra

#endif // ASTRA_CUDA
