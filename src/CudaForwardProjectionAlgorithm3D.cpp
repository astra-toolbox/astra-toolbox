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

#include "astra/CudaForwardProjectionAlgorithm3D.h"

#ifdef ASTRA_CUDA

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"

#include "astra/CompositeGeometryManager.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/astra3d.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaForwardProjectionAlgorithm3D::CCudaForwardProjectionAlgorithm3D() 
{
	m_bIsInitialized = false;
	m_iGPUIndex = -1;
	m_iDetectorSuperSampling = 1;
	m_pProjector = 0;
	m_pProjections = 0;
	m_pVolume = 0;

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaForwardProjectionAlgorithm3D::~CCudaForwardProjectionAlgorithm3D() 
{

}

//---------------------------------------------------------------------------------------
void CCudaForwardProjectionAlgorithm3D::initializeFromProjector()
{
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to FP3D_CUDA");
		}
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaForwardProjectionAlgorithm3D::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaForwardProjectionAlgorithm3D", this, _cfg);

	// projector
	m_pProjector = 0;
	int id = -1;
	if (CR.has("ProjectorId")) {
		CR.getID("ProjectorId", id);
		m_pProjector = CProjector3DManager::getSingleton().get(id);
		if (!m_pProjector) {
			ASTRA_WARN("Optional parameter ProjectorId is not a valid id");
		}
	}

	bool ok = true;

	// sinogram data
	ok &= CR.getRequiredID("ProjectionDataId", id);
	m_pProjections = dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(id));

	// reconstruction data
	ok &= CR.getRequiredID("VolumeDataId", id);
	m_pVolume = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));

	if (!ok)
		return false;

	initializeFromProjector();

	// Deprecated options
	ok &= CR.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling, m_iDetectorSuperSampling);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);
	if (!ok)
		return false;

	// success
	m_bIsInitialized = check();

	if (!m_bIsInitialized)
		return false;

	return true;	
}


bool CCudaForwardProjectionAlgorithm3D::initialize(CProjector3D* _pProjector, 
                                  CFloat32ProjectionData3D* _pProjections, 
                                  CFloat32VolumeData3D* _pVolume,
                                  int _iGPUindex, int _iDetectorSuperSampling)
{
	m_pProjector = _pProjector;
	
	// required classes
	m_pProjections = _pProjections;
	m_pVolume = _pVolume;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		// TODO: Report
		m_iDetectorSuperSampling = _iDetectorSuperSampling;
		m_iGPUIndex = _iGPUindex;
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}

	// success
	m_bIsInitialized = check();

	if (!m_bIsInitialized)
		return false;

	return true;
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaForwardProjectionAlgorithm3D::check() 
{
	// check pointers
	//ASTRA_CONFIG_CHECK(m_pProjector, "Reconstruction2D", "Invalid Projector Object.");
	ASTRA_CONFIG_CHECK(m_pProjections, "FP3D_CUDA", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pVolume, "FP3D_CUDA", "Invalid Volume Data Object.");

	// check initializations
	//ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "Reconstruction2D", "Projector Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pProjections->isInitialized(), "FP3D_CUDA", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pVolume->isInitialized(), "FP3D_CUDA", "Volume Data Object Not Initialized.");

	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "FP3D_CUDA", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FP3D_CUDA", "GPUIndex must be a non-negative integer.");

	// check compatibility between projector and data classes
//	ASTRA_CONFIG_CHECK(m_pSinogram->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "SIRT_CUDA", "Projection Data not compatible with the specified Projector.");
//	ASTRA_CONFIG_CHECK(m_pReconstruction->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "SIRT_CUDA", "Reconstruction Data not compatible with the specified Projector.");

	// todo: turn some of these back on

// 	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "SIRT_CUDA", "ProjectionGeometry not specified.");
// 	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "SIRT_CUDA", "ProjectionGeometry not initialized.");
// 	ASTRA_CONFIG_CHECK(m_pReconstructionGeometry, "SIRT_CUDA", "ReconstructionGeometry not specified.");
// 	ASTRA_CONFIG_CHECK(m_pReconstructionGeometry->isInitialized(), "SIRT_CUDA", "ReconstructionGeometry not initialized.");

	// check dimensions
	//ASTRA_CONFIG_CHECK(m_pSinogram->getAngleCount() == m_pProjectionGeometry->getProjectionAngleCount(), "SIRT_CUDA", "Sinogram data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pSinogram->getDetectorCount() == m_pProjectionGeometry->getDetectorCount(), "SIRT_CUDA", "Sinogram data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pReconstruction->getWidth() == m_pReconstructionGeometry->getGridColCount(), "SIRT_CUDA", "Reconstruction data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pReconstruction->getHeight() == m_pReconstructionGeometry->getGridRowCount(), "SIRT_CUDA", "Reconstruction data object size mismatch.");
	
	// check restrictions
	// TODO: check restrictions built into cuda code

	// success
	m_bIsInitialized = true;
	return true;
}


void CCudaForwardProjectionAlgorithm3D::setGPUIndex(int _iGPUIndex)
{
	m_iGPUIndex = _iGPUIndex;
}

//----------------------------------------------------------------------------------------
// Run
bool CCudaForwardProjectionAlgorithm3D::run(int)
{
	// check initialized
	assert(m_bIsInitialized);

	CCompositeGeometryManager cgm;

	return cgm.doFP(m_pProjector, m_pVolume, m_pProjections);
}


}

#endif
