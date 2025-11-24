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

#include "astra/CudaSirtAlgorithm3D.h"

#include "astra/AstraObjectManager.h"

#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/VolumeGeometry3D.h"
#include "astra/CudaProjector3D.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/astra3d.h"
#include "astra/cuda/3d/mem3d.h"
#include "astra/cuda/3d/arith3d.h"

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaSirtAlgorithm3D::CCudaSirtAlgorithm3D() 
	: m_bBuffersInitialized(false),
	  m_iGPUIndex(-1),
	  m_fRelaxation(1.0f),
	  D_projData(nullptr),
	  D_volData(nullptr),
	  D_tmpProjData(nullptr),
	  D_tmpVolData(nullptr),
	  D_lineWeight(nullptr),
	  D_pixelWeight(nullptr),
	  D_projMaskData(nullptr),
	  D_volMaskData(nullptr)
{

}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaSirtAlgorithm3D::CCudaSirtAlgorithm3D(CProjector3D* _pProjector,
                                           CFloat32ProjectionData3D* _pProjectionData,
                                           CFloat32VolumeData3D* _pReconstruction)
	: CCudaSirtAlgorithm3D()
{
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSirtAlgorithm3D::~CCudaSirtAlgorithm3D() 
{
	if (D_projData) {
		astraCUDA3d::freeGPUMemory(D_projData);
		delete D_projData;
	}
	if (D_volData) {
		astraCUDA3d::freeGPUMemory(D_volData);
		delete D_volData;
	}
	if (D_tmpProjData) {
		astraCUDA3d::freeGPUMemory(D_tmpProjData);
		delete D_tmpProjData;
	}
	if (D_tmpVolData) {
		astraCUDA3d::freeGPUMemory(D_tmpVolData);
		delete D_tmpVolData;
	}
	if (D_projMaskData) {
		astraCUDA3d::freeGPUMemory(D_projMaskData);
		delete D_projMaskData;
	}
	if (D_volMaskData) {
		astraCUDA3d::freeGPUMemory(D_volMaskData);
		delete D_volMaskData;
	}
	if (D_lineWeight) {
		astraCUDA3d::freeGPUMemory(D_lineWeight);
		delete D_lineWeight;
	}
	if (D_pixelWeight) {
		astraCUDA3d::freeGPUMemory(D_pixelWeight);
		delete D_pixelWeight;
	}
}


//---------------------------------------------------------------------------------------
// Check
bool CCudaSirtAlgorithm3D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm3D::_check(), "SIRT3D", "Error in ReconstructionAlgorithm3D initialization");


	return true;
}

//----------------------------------------------------------------------------------------
void CCudaSirtAlgorithm3D::initializeFromProjector()
{
	m_params.iRaysPerVoxelDim = 1;
	m_params.iRaysPerDetDim = 1;
	m_params.projKernel = astraCUDA3d::ker3d_default;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to SIRT3D_CUDA");
		}
	} else {
		m_iGPUIndex = pCudaProjector->getGPUIndex();

		m_params.iRaysPerVoxelDim = pCudaProjector->getVoxelSuperSampling();
		m_params.iRaysPerDetDim = pCudaProjector->getDetectorSuperSampling();
		m_params.projKernel = pCudaProjector->getProjectionKernel();
	}
}

//--------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSirtAlgorithm3D::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaSirtAlgorithm3D", this, _cfg);

	// initialization of parent class
	if (!CReconstructionAlgorithm3D::initialize(_cfg)) {
		return false;
	}

	bool ok = true;

	ok &= CR.getOptionNumerical("Relaxation", m_fRelaxation, 1.0f);

	initializeFromProjector();

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
		ASTRA_CONFIG_CHECK(false, "SIRT3D_CUDA",
						   "Detector/voxel supersampling is not supported for cyl_cone_vec geometry.");
	}

	m_bBuffersInitialized = false;

	if (!setupGeometry())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaSirtAlgorithm3D::initialize(CProjector3D* _pProjector,
                                      CFloat32ProjectionData3D* _pSinogram,
                                      CFloat32VolumeData3D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	m_fRelaxation = 1.0f;

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	initializeFromProjector();

	m_bBuffersInitialized = false;

	if (!setupGeometry())
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//--------------------------------------------------------------------------------------

bool CCudaSirtAlgorithm3D::setupGeometry()
{
	m_geometry = astra::convertAstraGeometry(&m_pReconstruction->getGeometry(), &m_pSinogram->getGeometry());
	m_params.volScale = m_geometry.getVolScale();

	return m_geometry.isValid();
}

bool CCudaSirtAlgorithm3D::precomputeWeights()
{
	bool ok = true;

	astraCUDA3d::zeroGPUMemory(D_lineWeight);
	if (m_bUseReconstructionMask) {
		callFP(D_volMaskData, D_lineWeight, 1.0f);
	} else {
		astraCUDA3d::processVol3D<astraCUDA3d::opSet>(D_tmpVolData, 1.0f);
		callFP(D_tmpVolData, D_lineWeight, 1.0f);
	}
	astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(D_lineWeight);

	if (m_bUseSinogramMask) {
		// scale line weights with sinogram mask to zero out masked sinogram pixels
		astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_lineWeight, D_projMaskData);
	}

	astraCUDA3d::zeroGPUMemory(D_pixelWeight);

	if (m_bUseSinogramMask) {
		callBP(D_pixelWeight, D_projMaskData, 1.0f);
	} else {
		astraCUDA3d::processVol3D<astraCUDA3d::opSet>(D_tmpProjData, 1.0f);
		callBP(D_pixelWeight, D_tmpProjData, 1.0f);
	}
	astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(D_pixelWeight);

	if (m_bUseReconstructionMask) {
		// scale pixel weights with mask to zero out masked pixels
		astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_pixelWeight, D_volMaskData);
	}
	astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_pixelWeight, m_fRelaxation);

	return ok;
}

//----------------------------------------------------------------------------------------
// Iterate

// TODO: Centralize this somehow
static void freeGPUMem(CData3D*& ptr)
{
	if (ptr) {
		astraCUDA3d::freeGPUMemory(ptr);
		delete ptr;
		ptr = nullptr;
	}
}

bool CCudaSirtAlgorithm3D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	bool ok = true;

	if (m_iGPUIndex != -1)
		astraCUDA3d::setGPUIndex(m_iGPUIndex);

	if (!m_bBuffersInitialized) {
		std::array<int, 3> volDims = m_pReconstruction->getShape();
		std::array<int, 3> projDims = m_pSinogram->getShape();

		const astra::CProjectionGeometry3D &pProjGeom = m_pSinogram->getGeometry();
		const astra::CVolumeGeometry3D &pVolGeom = m_pReconstruction->getGeometry();

		m_geometry = convertAstraGeometry(&pVolGeom, &pProjGeom);
		m_params.volScale = m_geometry.getVolScale();

		CDataStorage *s;

		s = astraCUDA3d::allocateGPUMemory(volDims[0], volDims[1], volDims[2], astraCUDA3d::INIT_ZERO);
		if (!s) {
			return false;
		}
		D_volData = new CData3D(volDims[0], volDims[1], volDims[2], s);

		s = astraCUDA3d::allocateGPUMemory(volDims[0], volDims[1], volDims[2], astraCUDA3d::INIT_ZERO);
		if (!s) {
			freeGPUMem(D_volData);
			return false;
		}
		D_pixelWeight = new CData3D(volDims[0], volDims[1], volDims[2], s);

		s = astraCUDA3d::allocateGPUMemory(volDims[0], volDims[1], volDims[2], astraCUDA3d::INIT_ZERO);
		if (!s) {
			freeGPUMem(D_volData);
			freeGPUMem(D_pixelWeight);
			return false;
		}
		D_tmpVolData = new CData3D(volDims[0], volDims[1], volDims[2], s);

		if (m_bUseReconstructionMask) {
			s = astraCUDA3d::allocateGPUMemory(volDims[0], volDims[1], volDims[2], astraCUDA3d::INIT_ZERO);
			if (!s) {
				freeGPUMem(D_volData);
				freeGPUMem(D_pixelWeight);
				freeGPUMem(D_tmpVolData);
				return false;
			}
			D_volMaskData = new CData3D(volDims[0], volDims[1], volDims[2], s);
		}

		s = astraCUDA3d::allocateGPUMemory(projDims[0], projDims[1], projDims[2], astraCUDA3d::INIT_ZERO);
		if (!s) {
			freeGPUMem(D_volData);
			freeGPUMem(D_pixelWeight);
			freeGPUMem(D_tmpVolData);
			freeGPUMem(D_volMaskData);
			return false;
		}
		D_projData = new CData3D(projDims[0], projDims[1], projDims[2], s);

		s = astraCUDA3d::allocateGPUMemory(projDims[0], projDims[1], projDims[2], astraCUDA3d::INIT_ZERO);
		if (!s) {
			freeGPUMem(D_volData);
			freeGPUMem(D_pixelWeight);
			freeGPUMem(D_tmpVolData);
			freeGPUMem(D_volMaskData);
			freeGPUMem(D_projData);
			return false;
		}
		D_tmpProjData = new CData3D(projDims[0], projDims[1], projDims[2], s);

		s = astraCUDA3d::allocateGPUMemory(projDims[0], projDims[1], projDims[2], astraCUDA3d::INIT_ZERO);
		if (!s) {
			freeGPUMem(D_volData);
			freeGPUMem(D_pixelWeight);
			freeGPUMem(D_tmpVolData);
			freeGPUMem(D_volMaskData);
			freeGPUMem(D_projData);
			freeGPUMem(D_tmpProjData);
			return false;
		}
		D_lineWeight = new CData3D(projDims[0], projDims[1], projDims[2], s);

		if (m_bUseSinogramMask) {
			s = astraCUDA3d::allocateGPUMemory(projDims[0], projDims[1], projDims[2], astraCUDA3d::INIT_ZERO);
			if (!s) {
				freeGPUMem(D_volData);
				freeGPUMem(D_pixelWeight);
				freeGPUMem(D_tmpVolData);
				freeGPUMem(D_volMaskData);
				freeGPUMem(D_projData);
				freeGPUMem(D_tmpProjData);
				freeGPUMem(D_lineWeight);
				return false;
			}
			D_projMaskData = new CData3D(projDims[0], projDims[1], projDims[2], s);
		}


		// We can't precompute lineWeights and pixelWeights when using a mask
		if (!m_bUseReconstructionMask && !m_bUseSinogramMask)
			ok &= precomputeWeights();

		if (!ok) {
			freeGPUMem(D_volData);
			freeGPUMem(D_pixelWeight);
			freeGPUMem(D_tmpVolData);
			freeGPUMem(D_volMaskData);
			freeGPUMem(D_projData);
			freeGPUMem(D_tmpProjData);
			freeGPUMem(D_lineWeight);
			freeGPUMem(D_projMaskData);
			return false;
		}

		m_bBuffersInitialized = true;
	}

	ASTRA_ASSERT(m_pSinogram->isFloat32Memory());

	ok &= astraCUDA3d::copyToGPUMemory(m_pSinogram, D_projData);

	if (m_bUseReconstructionMask) {
		ASTRA_ASSERT(m_pReconstructionMask->isFloat32Memory());
		ok &= astraCUDA3d::copyToGPUMemory(m_pReconstructionMask, D_volMaskData);
	}
	if (m_bUseSinogramMask) {
		ASTRA_ASSERT(m_pSinogramMask->isFloat32Memory());
		ok &= astraCUDA3d::copyToGPUMemory(m_pSinogramMask, D_projMaskData);
	}

	ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
	ok &= astraCUDA3d::copyToGPUMemory(m_pReconstruction, D_volData);

	if (!ok)
		return false;

	if (m_bUseReconstructionMask || m_bUseSinogramMask)
		ok &= precomputeWeights();

	if (!ok)
		return false;

	for (int iter = 0; iter < _iNrIterations && !astra::shouldAbort(); ++iter) {
		// TODO: Error checking in this loop

		// copy projection data to tmp buffer
		astraCUDA3d::assignGPUMemory(D_tmpProjData, D_projData);

		// do FP, subtracting it from projection data
		if (m_bUseReconstructionMask) {
			astraCUDA3d::assignGPUMemory(D_tmpVolData, D_volData);
			astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_tmpVolData, D_volMaskData);
			callFP(D_tmpVolData, D_tmpProjData, -1.0f);
		} else {
			callFP(D_volData, D_tmpProjData, -1.0f);
		}

		astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_tmpProjData, D_lineWeight);

		astraCUDA3d::zeroGPUMemory(D_tmpVolData);

		callBP(D_tmpVolData, D_tmpProjData, 1.0f);

		// pixel weights also contain the volume mask and relaxation factor
		astraCUDA3d::processVol3D<astraCUDA3d::opAddMul>(D_volData, D_tmpVolData, D_pixelWeight);

		if (m_bUseMinConstraint)
			astraCUDA3d::processVol3D<astraCUDA3d::opClampMin>(D_volData, m_fMinValue);
		if (m_bUseMaxConstraint)
			astraCUDA3d::processVol3D<astraCUDA3d::opClampMax>(D_volData, m_fMaxValue);
	}

	ok &= astraCUDA3d::copyFromGPUMemory(m_pReconstruction, D_volData);

	return ok;
}
//----------------------------------------------------------------------------------------
bool CCudaSirtAlgorithm3D::getResidualNorm(float32& _fNorm)
{
	if (!m_bIsInitialized || !m_bBuffersInitialized)
		return false;

	if (m_iGPUIndex != -1)
		astraCUDA3d::setGPUIndex(m_iGPUIndex);

	bool ok = true;

	// copy sinogram to projection data
	ok &= astraCUDA3d::assignGPUMemory(D_tmpProjData, D_projData);

	if (!ok)
		return false;

	// do FP, subtracting projection from sinogram
	if (m_bUseReconstructionMask) {
		ok &= astraCUDA3d::assignGPUMemory(D_tmpVolData, D_volData);
		ok &= astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_tmpVolData, D_volMaskData);
		ok &= callFP(D_tmpVolData, D_tmpProjData, -1.0f);
	} else {
		ok &= callFP(D_volData, D_tmpProjData, -1.0f);
	}

	if (!ok)
		return false;

	float s = astraCUDA3d::dotProduct3D(D_tmpProjData);
	_fNorm = sqrt(s);

	return true;
}
//----------------------------------------------------------------------------------------
bool CCudaSirtAlgorithm3D::callFP(const CData3D *D_vol, CData3D *D_proj, float fScale)
{
	astraCUDA3d::SProjectorParams3D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA3d::FP(D_proj, D_vol, m_geometry, p);
}

bool CCudaSirtAlgorithm3D::callBP(CData3D *D_vol, const CData3D *D_proj, float fScale)
{
	astraCUDA3d::SProjectorParams3D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA3d::BP(D_proj, D_vol, m_geometry, p);
}



} // namespace astra
