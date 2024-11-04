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

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaSirtAlgorithm3D::CCudaSirtAlgorithm3D() 
{
	m_bIsInitialized = false;
	m_pSirt = 0;
	m_iGPUIndex = -1;
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_fLambda = 1.0f;
}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaSirtAlgorithm3D::CCudaSirtAlgorithm3D(CProjector3D* _pProjector,
								   CFloat32ProjectionData3D* _pProjectionData,
								   CFloat32VolumeData3D* _pReconstruction)
{
	_clear();
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSirtAlgorithm3D::~CCudaSirtAlgorithm3D() 
{
	delete m_pSirt;
	m_pSirt = 0;

	CReconstructionAlgorithm3D::_clear();
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
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to SIRT3D_CUDA");
		}
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}

}

//--------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSirtAlgorithm3D::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaSirtAlgorithm3D", this, _cfg);


	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CReconstructionAlgorithm3D::initialize(_cfg)) {
		return false;
	}

	bool ok = true;

	ok &= CR.getOptionNumerical("Relaxation", m_fLambda, 1.0f);

	initializeFromProjector();

	// Deprecated options
	ok &= CR.getOptionInt("VoxelSuperSampling", m_iVoxelSuperSampling, m_iVoxelSuperSampling);
	ok &= CR.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling, m_iDetectorSuperSampling);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, m_iGPUIndex);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, m_iGPUIndex);
	if (!ok)
		return false;


	m_pSirt = new AstraSIRT3d();

	m_bAstraSIRTInit = false;


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
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	m_fLambda = 1.0f;

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	m_pSirt = new AstraSIRT3d;

	m_bAstraSIRTInit = false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Iterate
bool CCudaSirtAlgorithm3D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	const CProjectionGeometry3D &projgeom = m_pSinogram->getGeometry();
	const CVolumeGeometry3D &volgeom = m_pReconstruction->getGeometry();

	bool ok = true;

	if (!m_bAstraSIRTInit) {

		ok &= m_pSirt->setGPUIndex(m_iGPUIndex);

		ok &= m_pSirt->setGeometry(&volgeom, &projgeom);

		ok &= m_pSirt->enableSuperSampling(m_iVoxelSuperSampling, m_iDetectorSuperSampling);

		if (m_bUseReconstructionMask)
			ok &= m_pSirt->enableVolumeMask();
		if (m_bUseSinogramMask)
			ok &= m_pSirt->enableSinogramMask();

		ok &= m_pSirt->setRelaxation(m_fLambda);

		ASTRA_ASSERT(ok);

		ok &= m_pSirt->init();

		ASTRA_ASSERT(ok);

		m_bAstraSIRTInit = true;

	}

	ASTRA_ASSERT(m_pSinogram->isFloat32Memory());

	ok = m_pSirt->setSinogram(m_pSinogram->getFloat32Memory(), m_pSinogram->getGeometry().getDetectorColCount());

	ASTRA_ASSERT(ok);

	if (m_bUseReconstructionMask) {
		ASTRA_ASSERT(m_pReconstructionMask->isFloat32Memory());
		ok &= m_pSirt->setVolumeMask(m_pReconstructionMask->getFloat32Memory(), volgeom.getGridColCount());
	}
	if (m_bUseSinogramMask) {
		ASTRA_ASSERT(m_pSinogramMask->isFloat32Memory());
		ok &= m_pSirt->setSinogramMask(m_pSinogramMask->getFloat32Memory(), m_pSinogramMask->getGeometry().getDetectorColCount());
	}

	ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
	ok &= m_pSirt->setStartReconstruction(m_pReconstruction->getFloat32Memory(),
	                                      volgeom.getGridColCount());

	ASTRA_ASSERT(ok);

	if (m_bUseMinConstraint)
		ok &= m_pSirt->setMinConstraint(m_fMinValue);
	if (m_bUseMaxConstraint)
		ok &= m_pSirt->setMaxConstraint(m_fMaxValue);

	ok &= m_pSirt->iterate(_iNrIterations);
	ASTRA_ASSERT(ok);

	ok &= m_pSirt->getReconstruction(m_pReconstruction->getFloat32Memory(),
	                                 volgeom.getGridColCount());
	ASTRA_ASSERT(ok);

	return ok;
}
//----------------------------------------------------------------------------------------
bool CCudaSirtAlgorithm3D::getResidualNorm(float32& _fNorm)
{
	if (!m_bIsInitialized || !m_pSirt)
		return false;

	_fNorm = m_pSirt->computeDiffNorm();

	return true;
}


} // namespace astra
