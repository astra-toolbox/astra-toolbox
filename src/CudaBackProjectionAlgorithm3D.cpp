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

#include "astra/CudaBackProjectionAlgorithm3D.h"

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
CCudaBackProjectionAlgorithm3D::CCudaBackProjectionAlgorithm3D() 
{
	m_bIsInitialized = false;
	m_iGPUIndex = -1;
	m_iVoxelSuperSampling = 1;
	m_bSIRTWeighting = false;
}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaBackProjectionAlgorithm3D::CCudaBackProjectionAlgorithm3D(CProjector3D* _pProjector, 
								   CFloat32ProjectionData3D* _pProjectionData, 
								   CFloat32VolumeData3D* _pReconstruction)
{
	_clear();
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaBackProjectionAlgorithm3D::~CCudaBackProjectionAlgorithm3D() 
{
	CReconstructionAlgorithm3D::_clear();
}


//---------------------------------------------------------------------------------------
// Check
bool CCudaBackProjectionAlgorithm3D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm3D::_check(), "BP3D_CUDA", "Error in ReconstructionAlgorithm3D initialization");


	return true;
}

//---------------------------------------------------------------------------------------
void CCudaBackProjectionAlgorithm3D::initializeFromProjector()
{
	m_iVoxelSuperSampling = 1;
	m_iGPUIndex = -1;

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to BP3D_CUDA");
		}
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_iGPUIndex = pCudaProjector->getGPUIndex();
	}

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaBackProjectionAlgorithm3D::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaBackProjectionAlgorithm3D", this, _cfg);

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
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, m_iGPUIndex);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, m_iGPUIndex);

	ok &= CR.getOptionBool("SIRTWeighting", m_bSIRTWeighting, false);

	if (!ok)
		return false;

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaBackProjectionAlgorithm3D::initialize(CProjector3D* _pProjector, 
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

	m_bSIRTWeighting = false;

	initializeFromProjector();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Iterate
bool CCudaBackProjectionAlgorithm3D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	CFloat32ProjectionData3D* pSinoMem = dynamic_cast<CFloat32ProjectionData3D*>(m_pSinogram);
	ASTRA_ASSERT(pSinoMem);
	CFloat32VolumeData3D* pReconMem = dynamic_cast<CFloat32VolumeData3D*>(m_pReconstruction);
	ASTRA_ASSERT(pReconMem);

	const CProjectionGeometry3D& projgeom = pSinoMem->getGeometry();
	const CVolumeGeometry3D& volgeom = pReconMem->getGeometry();

	if (m_bSIRTWeighting) {
		ASTRA_ASSERT(m_pSinogram->isFloat32Memory());
		ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());
		return astraCudaBP_SIRTWeighted(m_pReconstruction->getFloat32Memory(),
		                                m_pSinogram->getFloat32Memory(),
		                                &volgeom, &projgeom,
		                                m_iGPUIndex, m_iVoxelSuperSampling);
	} else {
		CCompositeGeometryManager cgm;

		return cgm.doBP(m_pProjector, pReconMem, pSinoMem);
	}

}


} // namespace astra
