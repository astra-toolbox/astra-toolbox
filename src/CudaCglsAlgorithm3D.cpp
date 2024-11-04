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

#include "astra/CudaCglsAlgorithm3D.h"

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/VolumeGeometry3D.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/astra3d.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaCglsAlgorithm3D::CCudaCglsAlgorithm3D() 
{
	m_bIsInitialized = false;
	m_pCgls = 0;
	m_iGPUIndex = -1;
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaCglsAlgorithm3D::CCudaCglsAlgorithm3D(CProjector3D* _pProjector,
								   CFloat32ProjectionData3D* _pProjectionData,
								   CFloat32VolumeData3D* _pReconstruction)
{
	_clear();
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaCglsAlgorithm3D::~CCudaCglsAlgorithm3D() 
{
	delete m_pCgls;
	m_pCgls = 0;

	CReconstructionAlgorithm3D::_clear();
}


//---------------------------------------------------------------------------------------
// Check
bool CCudaCglsAlgorithm3D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm3D::_check(), "CGLS3D", "Error in ReconstructionAlgorithm3D initialization");


	return true;
}

//---------------------------------------------------------------------------------------
void CCudaCglsAlgorithm3D::initializeFromProjector()
{
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to CGLS3D_CUDA");
		}
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaCglsAlgorithm3D::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaCglsAlgorithm3D", this, _cfg);


	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CReconstructionAlgorithm3D::initialize(_cfg)) {
		return false;
	}

	initializeFromProjector();

	bool ok = true;

	// Deprecated options
	ok &= CR.getOptionInt("VoxelSuperSampling", m_iVoxelSuperSampling, m_iVoxelSuperSampling);
	ok &= CR.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling, m_iDetectorSuperSampling);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, m_iGPUIndex);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, m_iGPUIndex);
	if (!ok)
		return false;


	m_pCgls = new AstraCGLS3d();

	m_bAstraCGLSInit = false;


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
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	initializeFromProjector();

	m_pCgls = new AstraCGLS3d;

	m_bAstraCGLSInit = false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Iterate
bool CCudaCglsAlgorithm3D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	const CProjectionGeometry3D& projgeom = m_pSinogram->getGeometry();
	const CVolumeGeometry3D& volgeom = m_pReconstruction->getGeometry();

	bool ok = true;

	if (!m_bAstraCGLSInit) {

		ok &= m_pCgls->setGPUIndex(m_iGPUIndex);

		ok &= m_pCgls->setGeometry(&volgeom, &projgeom);

		ok &= m_pCgls->enableSuperSampling(m_iVoxelSuperSampling, m_iDetectorSuperSampling);

		if (m_bUseReconstructionMask)
			ok &= m_pCgls->enableVolumeMask();
#if 0
		if (m_bUseSinogramMask)
			ok &= m_pCgls->enableSinogramMask();
#endif

		ASTRA_ASSERT(ok);

		ok &= m_pCgls->init();

		ASTRA_ASSERT(ok);

		m_bAstraCGLSInit = true;

	}

	ASTRA_ASSERT(m_pSinogram->isFloat32Memory());

	ok = m_pCgls->setSinogram(m_pSinogram->getFloat32Memory(), m_pSinogram->getGeometry().getDetectorColCount());

	ASTRA_ASSERT(ok);

	if (m_bUseReconstructionMask) {
		ASTRA_ASSERT(m_pReconstructionMask->isFloat32Memory());
		ok &= m_pCgls->setVolumeMask(m_pReconstructionMask->getFloat32Memory(), volgeom.getGridColCount());
	}
#if 0
	if (m_bUseSinogramMask) {
		CFloat32ProjectionData3DMemory* pSMaskMem = dynamic_cast<CFloat32ProjectionData3DMemory*>(m_pSinogramMask);
		ASTRA_ASSERT(m_pSinogramMask->isFloat32Memory());
		ok &= m_pCgls->setSinogramMask(m_pSinogramMask->getFloat32Memory(), m_pSinogramMask->getGeometry()->getDetectorColCount());
	}
#endif

	ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
	ok &= m_pCgls->setStartReconstruction(m_pReconstruction->getFloat32Memory(),
	                                      volgeom.getGridColCount());

	ASTRA_ASSERT(ok);

#if 0
	if (m_bUseMinConstraint)
		ok &= m_pCgls->setMinConstraint(m_fMinValue);
	if (m_bUseMaxConstraint)
		ok &= m_pCgls->setMaxConstraint(m_fMaxValue);
#endif

	ok &= m_pCgls->iterate(_iNrIterations);
	ASTRA_ASSERT(ok);

	ok &= m_pCgls->getReconstruction(m_pReconstruction->getFloat32Memory(),
	                                 volgeom.getGridColCount());
	ASTRA_ASSERT(ok);

	return ok;
}
//----------------------------------------------------------------------------------------
bool CCudaCglsAlgorithm3D::getResidualNorm(float32& _fNorm)
{
	if (!m_bIsInitialized || !m_pCgls)
		return false;

	_fNorm = m_pCgls->computeDiffNorm();

	return true;
}



} // namespace astra
