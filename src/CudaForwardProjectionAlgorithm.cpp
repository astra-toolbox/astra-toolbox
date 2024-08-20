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

#include "astra/CudaForwardProjectionAlgorithm.h"

#ifdef ASTRA_CUDA

#include "astra/cuda/2d/astra.h"

#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "astra/AstraObjectManager.h"
#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/Float32ProjectionData2D.h"
#include "astra/Float32VolumeData2D.h"
#include "astra/CudaProjector2D.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaForwardProjectionAlgorithm::type = "FP_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaForwardProjectionAlgorithm::CCudaForwardProjectionAlgorithm() 
{
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaForwardProjectionAlgorithm::~CCudaForwardProjectionAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
void CCudaForwardProjectionAlgorithm::initializeFromProjector()
{
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;

	// Projector
	CCudaProjector2D* pCudaProjector = dynamic_cast<CCudaProjector2D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector2D passed to FP_CUDA");
		}
	} else {
		m_iDetectorSuperSampling = pCudaProjector->getDetectorSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaForwardProjectionAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaForwardProjectionAlgorithm", this, _cfg);

	// Projector
	m_pProjector = 0;
	XMLNode node = _cfg.self.getSingleNode("ProjectorId");
	if (node) {
		int id = StringUtil::stringToInt(node.getContent(), -1);
		m_pProjector = CProjector2DManager::getSingleton().get(id);
	}
	CC.markNodeParsed("ProjectorId");


	
	// sinogram data
	node = _cfg.self.getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "FP_CUDA", "No ProjectionDataId tag specified.");
	int id = StringUtil::stringToInt(node.getContent(), -1);
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("ProjectionDataId");

	// volume data
	node = _cfg.self.getSingleNode("VolumeDataId");
	ASTRA_CONFIG_CHECK(node, "FP_CUDA", "No VolumeDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pVolume = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("VolumeDataId");

	initializeFromProjector();

	// Deprecated options
	try {
		m_iDetectorSuperSampling = _cfg.self.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "FP_CUDA", "Supersampling options must be integers.");
	}
	CC.markOptionParsed("DetectorSuperSampling");
	// GPU number
	try {
		m_iGPUIndex = _cfg.self.getOptionInt("GPUindex", -1);
		m_iGPUIndex = _cfg.self.getOptionInt("GPUIndex", m_iGPUIndex);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "FP_CUDA", "GPUIndex must be an integer.");
	}
	CC.markOptionParsed("GPUIndex");
	if (!_cfg.self.hasOption("GPUIndex"))
		CC.markOptionParsed("GPUindex");



	// return success
	return check();
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaForwardProjectionAlgorithm::initialize(CProjector2D* _pProjector,
												 CFloat32VolumeData2D* _pVolume,
												 CFloat32ProjectionData2D* _pSinogram)
{
	// store classes
	m_pProjector = _pProjector;
	m_pVolume = _pVolume;
	m_pSinogram = _pSinogram;

	initializeFromProjector();

	// return success
	return check();
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaForwardProjectionAlgorithm::check() 
{
	// check pointers
	ASTRA_CONFIG_CHECK(m_pSinogram, "FP_CUDA", "No valid projection data object found.");
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "FP_CUDA", "Projection data not initialized.");
	ASTRA_CONFIG_CHECK(m_pVolume, "FP_CUDA", "No valid volume data object found.");
	ASTRA_CONFIG_CHECK(m_pVolume->isInitialized(), "FP_CUDA", "Volume data not initialized.");

	// check restrictions
	//int iImageSideBlocks = m_pReconstructionGeometry->getGridColCount() / G_BLOCKIMAGESIZE;
	//ASTRA_CONFIG_CHECK((iImageSideBlocks * G_BLOCKIMAGESIZE) == m_pVolume->getWidth(), "FP_CUDA", "Volume Width must be a multiple of G_BLOCKIMAGESIZE");
	//ASTRA_CONFIG_CHECK((iImageSideBlocks * G_BLOCKIMAGESIZE) == m_pVolume->getHeight(), "FP_CUDA", "Volume Height must be a multiple of G_BLOCKIMAGESIZE");
	//ASTRA_CONFIG_CHECK(m_pProjectionGeometry->getDetectorCount() == (m_pVolume->getWidth() * 3 / 2), "SIRT_CUDA", "Number of detectors must be 1.5 times the width of the image");

	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FP_CUDA", "GPUIndex must be a non-negative integer.");

	// success
	m_bIsInitialized = true;
	return true;
}

void CCudaForwardProjectionAlgorithm::setGPUIndex(int _iGPUIndex)
{
	m_iGPUIndex = _iGPUIndex;
}

//---------------------------------------------------------------------------------------
// Update Slices 
void CCudaForwardProjectionAlgorithm::updateSlice(CFloat32ProjectionData2D* _pSinogram,
                                     			  CFloat32VolumeData2D* _pVolume)
{
    m_pSinogram = _pSinogram;
    m_pVolume = _pVolume;
}

//----------------------------------------------------------------------------------------
// Run
void CCudaForwardProjectionAlgorithm::run(int)
{
	// check initialized
	assert(m_bIsInitialized);

	bool ok;

	const CVolumeGeometry2D* pVolGeom = m_pVolume->getGeometry();
	const CProjectionGeometry2D* pProjGeom = m_pSinogram->getGeometry();
	astraCUDA::SDimensions dims;

	ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);

	if (!ok)
		return;

	astraCUDA::SParProjection* pParProjs = 0;
	astraCUDA::SFanProjection* pFanProjs = 0;
	float fOutputScale = 1.0f;

	ok = convertAstraGeometry(pVolGeom, pProjGeom, pParProjs, pFanProjs, fOutputScale);
	if (!ok)
		return;

	if (pParProjs) {
		assert(!pFanProjs);

		ok = astraCudaFP(m_pVolume->getDataConst(), m_pSinogram->getData(),
		                 pVolGeom->getGridColCount(), pVolGeom->getGridRowCount(),
		                 pProjGeom->getProjectionAngleCount(),
		                 pProjGeom->getDetectorCount(),
		                 pParProjs,
		                 m_iDetectorSuperSampling, 1.0f * fOutputScale, m_iGPUIndex);

		delete[] pParProjs;

	} else {
		assert(pFanProjs);

		ok = astraCudaFanFP(m_pVolume->getDataConst(), m_pSinogram->getData(),
		                    pVolGeom->getGridColCount(), pVolGeom->getGridRowCount(),
		                    pProjGeom->getProjectionAngleCount(),
		                    pProjGeom->getDetectorCount(),
		                    pFanProjs,
		                    m_iDetectorSuperSampling, fOutputScale, m_iGPUIndex);

		delete[] pFanProjs;

	}

	ASTRA_ASSERT(ok);

}

} // namespace astra

#endif // ASTRA_CUDA
