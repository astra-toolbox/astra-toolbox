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

#include "astra/cuda/2d/mem2d.h"

#include <ctime>

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaReconstructionAlgorithm2D::CCudaReconstructionAlgorithm2D() 
	: m_iGPUIndex(-1)
{

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaReconstructionAlgorithm2D::~CCudaReconstructionAlgorithm2D() 
{

}

//---------------------------------------------------------------------------------------
void CCudaReconstructionAlgorithm2D::initializeFromProjector()
{
	m_params.iRaysPerDet = 1;
	m_params.iRaysPerPixelDim = 1;
	m_iGPUIndex = -1;

	// Projector
	CCudaProjector2D* pCudaProjector = dynamic_cast<CCudaProjector2D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector2D passed");
		}
	} else {
		m_params.iRaysPerDet = pCudaProjector->getDetectorSuperSampling();
		m_params.iRaysPerPixelDim = pCudaProjector->getVoxelSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaReconstructionAlgorithm2D::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaReconstructionAlgorithm2D", this, _cfg);

	if (!CReconstructionAlgorithm2D::initialize(_cfg))
		return false;

	initializeFromProjector();

	bool ok = true;

	// Deprecated options
	ok &= CR.getOptionUInt("PixelSuperSampling", m_params.iRaysPerPixelDim, m_params.iRaysPerPixelDim);
	ok &= CR.getOptionUInt("DetectorSuperSampling", m_params.iRaysPerDet, m_params.iRaysPerDet);

	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);

	if (!ok)
		return false;

	if (!setupGeometry())
		return false;

	return _check();
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaReconstructionAlgorithm2D::initialize(CProjector2D* _pProjector,
                                                CFloat32ProjectionData2D* _pSinogram, 
                                                CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);
	
	m_pProjector = _pProjector;
	
	// required classes
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	initializeFromProjector();

	setupGeometry();

	return _check();
}


//----------------------------------------------------------------------------------------
// Check
bool CCudaReconstructionAlgorithm2D::_check() 
{
	if (!CReconstructionAlgorithm2D::_check())
		return false;

	ASTRA_CONFIG_CHECK(m_params.iRaysPerDet >= 1, "CudaReconstructionAlgorithm2D", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_params.iRaysPerPixelDim >= 1, "CudaReconstructionAlgorithm2D", "PixelSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "CudaReconstructionAlgorithm2D", "GPUIndex must be a non-negative integer or -1.");
	ASTRA_CONFIG_CHECK(m_geometry.isValid(), "CudaReconstructionAlgorithm2D", "Invalid geometry type.");

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
	const CVolumeGeometry2D& volGeom = m_pReconstruction->getGeometry();
	const CProjectionGeometry2D& projGeom = m_pSinogram->getGeometry();

	m_geometry = convertAstraGeometry(&volGeom, &projGeom);

	m_params.fOutputScale = m_geometry.getOutputScale();

	return m_geometry.isValid();
}

bool CCudaReconstructionAlgorithm2D::callFP(const CData2D *D_vol, CData2D *D_proj, float fScale)
{
	astraCUDA::SProjectorParams2D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA::FP(D_proj, D_vol, m_geometry, p);
}

bool CCudaReconstructionAlgorithm2D::callBP(CData2D *D_vol, const CData2D *D_proj, float fScale)
{
	astraCUDA::SProjectorParams2D p = m_params;
	p.fOutputScale *= fScale;
	return astraCUDA::BP(D_proj, D_vol, m_geometry, p);
}

} // namespace astra

#endif // ASTRA_CUDA
