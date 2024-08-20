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

#include "astra/CudaReconstructionAlgorithm2D.h"

#include "astra/AstraObjectManager.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/CudaProjector2D.h"

#include "astra/Logging.h"

#include "astra/cuda/2d/algo.h"

#include <ctime>

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaReconstructionAlgorithm2D::CCudaReconstructionAlgorithm2D() 
{
	_clear();
}



//----------------------------------------------------------------------------------------
// Destructor
CCudaReconstructionAlgorithm2D::~CCudaReconstructionAlgorithm2D() 
{
	delete m_pAlgo;
	m_pAlgo = 0;
	m_bAlgoInit = false;
}

void CCudaReconstructionAlgorithm2D::clear()
{
	delete m_pAlgo;
	_clear();
}

void CCudaReconstructionAlgorithm2D::_clear()
{
	m_bIsInitialized = false;
	m_pAlgo = 0;
	m_bAlgoInit = false;
	CReconstructionAlgorithm2D::_clear();

	m_iGPUIndex = -1;
	m_iDetectorSuperSampling = 1;
	m_iPixelSuperSampling = 1;
}

//---------------------------------------------------------------------------------------
void CCudaReconstructionAlgorithm2D::initializeFromProjector()
{
	m_iPixelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	// Projector
	CCudaProjector2D* pCudaProjector = dynamic_cast<CCudaProjector2D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector2D passed");
		}
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iPixelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaReconstructionAlgorithm2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaReconstructionAlgorithm2D", this, _cfg);

	m_bIsInitialized = CReconstructionAlgorithm2D::initialize(_cfg);

	if (!m_bIsInitialized)
		return false;

	initializeFromProjector();

	// Deprecated options
	try {
		m_iDetectorSuperSampling = _cfg.self.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling);
		m_iPixelSuperSampling = _cfg.self.getOptionInt("PixelSuperSampling", m_iPixelSuperSampling);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaReconstructionAlgorithm2D", "Supersampling options must be integers.");
	}
	CC.markOptionParsed("DetectorSuperSampling");
	CC.markOptionParsed("PixelSuperSampling");

	// GPU number
	try {
		m_iGPUIndex = _cfg.self.getOptionInt("GPUindex", -1);
		m_iGPUIndex = _cfg.self.getOptionInt("GPUIndex", m_iGPUIndex);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaReconstructionAlgorithm2D", "GPUIndex must be an integer.");
	}
	CC.markOptionParsed("GPUIndex");
	if (!_cfg.self.hasOption("GPUIndex"))
		CC.markOptionParsed("GPUindex");

	return _check();
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaReconstructionAlgorithm2D::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram, 
                                     CFloat32VolumeData2D* _pReconstruction)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}
	
	m_pProjector = _pProjector;
	
	// required classes
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	initializeFromProjector();

	return _check();
}


//----------------------------------------------------------------------------------------
// Check
bool CCudaReconstructionAlgorithm2D::_check() 
{
	if (!CReconstructionAlgorithm2D::_check())
		return false;

	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "CudaReconstructionAlgorithm2D", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iPixelSuperSampling >= 1, "CudaReconstructionAlgorithm2D", "PixelSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "CudaReconstructionAlgorithm2D", "GPUIndex must be a non-negative integer or -1.");

	// check restrictions
	// TODO: check restrictions built into cuda code


	// success
	m_bIsInitialized = true;
	return true;
}

void CCudaReconstructionAlgorithm2D::setGPUIndex(int _iGPUIndex)
{
	m_iGPUIndex = _iGPUIndex;
}

bool CCudaReconstructionAlgorithm2D::setupGeometry()
{
	ASTRA_ASSERT(m_bIsInitialized);
	ASTRA_ASSERT(!m_bAlgoInit);

	bool ok;

	// TODO: Probably not the best place for this...
	ok = m_pAlgo->setGPUIndex(m_iGPUIndex);
	if (!ok) return false;

	const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();
	const CProjectionGeometry2D& projgeom = *m_pSinogram->getGeometry();

	ok = m_pAlgo->setGeometry(&volgeom, &projgeom);
	if (!ok) return false;

	ok = m_pAlgo->setSuperSampling(m_iDetectorSuperSampling, m_iPixelSuperSampling);
	if (!ok) return false;

	if (m_bUseReconstructionMask)
		ok &= m_pAlgo->enableVolumeMask();
	if (!ok) return false;
	if (m_bUseSinogramMask)
		ok &= m_pAlgo->enableSinogramMask();
	if (!ok) return false;

	ok &= m_pAlgo->init();
	if (!ok) return false;


	return true;
}

//----------------------------------------------------------------------------------------

void CCudaReconstructionAlgorithm2D::initCUDAAlgorithm()
{
	bool ok;

	ok = setupGeometry();
	ASTRA_ASSERT(ok);

	ok = m_pAlgo->allocateBuffers();
	ASTRA_ASSERT(ok);
}


//----------------------------------------------------------------------------------------
// Iterate
void CCudaReconstructionAlgorithm2D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	bool ok = true;
	const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();

	if (!m_bAlgoInit) {
		initCUDAAlgorithm();
		m_bAlgoInit = true;
	}

	ok = m_pAlgo->copyDataToGPU(m_pSinogram->getDataConst(), m_pSinogram->getGeometry()->getDetectorCount(),
	                            m_pReconstruction->getDataConst(), volgeom.getGridColCount(),
	                            m_bUseReconstructionMask ? m_pReconstructionMask->getDataConst() : 0, volgeom.getGridColCount(),
	                            m_bUseSinogramMask ? m_pSinogramMask->getDataConst() : 0, m_pSinogram->getGeometry()->getDetectorCount());

	ASTRA_ASSERT(ok);

	if (m_bUseMinConstraint) {
		bool ret = m_pAlgo->setMinConstraint(m_fMinValue);
		if (!ret) {
			ASTRA_WARN("This algorithm ignores MinConstraint");
		}
	}
	if (m_bUseMaxConstraint) {
		bool ret= m_pAlgo->setMaxConstraint(m_fMaxValue);
		if (!ret) {
			ASTRA_WARN("This algorithm ignores MaxConstraint");
		}
	}

	ok &= m_pAlgo->iterate(_iNrIterations);
	ASTRA_ASSERT(ok);

	ok &= m_pAlgo->getReconstruction(m_pReconstruction->getData(),
	                                 volgeom.getGridColCount());

	ASTRA_ASSERT(ok);
}

bool CCudaReconstructionAlgorithm2D::getResidualNorm(float32& _fNorm)
{
	if (!m_bIsInitialized || !m_pAlgo)
		return false;

	_fNorm = m_pAlgo->computeDiffNorm();

	return true;
}

} // namespace astra

#endif // ASTRA_CUDA
