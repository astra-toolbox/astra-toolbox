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

#include "astra/CudaBackProjectionAlgorithm3D.h"

#include "astra/AstraObjectManager.h"

#include "astra/CudaProjector3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/CompositeGeometryManager.h"

#include "astra/Logging.h"

#include "astra/cuda/3d/dims3d.h"
#include "astra/cuda/3d/arith3d.h"
#include "astra/cuda/3d/mem3d.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaBackProjectionAlgorithm3D::CCudaBackProjectionAlgorithm3D()
	: m_iVoxelSuperSampling(1), m_bSIRTWeighting(false)
{

}

//----------------------------------------------------------------------------------------
// Constructor with initialization
CCudaBackProjectionAlgorithm3D::CCudaBackProjectionAlgorithm3D(CProjector3D* _pProjector,
                                                               CFloat32ProjectionData3D* _pProjectionData,
                                                               CFloat32VolumeData3D* _pReconstruction)
	: CCudaBackProjectionAlgorithm3D()
{
	initialize(_pProjector, _pProjectionData, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaBackProjectionAlgorithm3D::~CCudaBackProjectionAlgorithm3D()
{

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
	m_GPUIndices.clear();

	CCudaProjector3D* pCudaProjector = dynamic_cast<CCudaProjector3D*>(m_pProjector);
	if (!pCudaProjector) {
		if (m_pProjector) {
			ASTRA_WARN("non-CUDA Projector3D passed to BP3D_CUDA");
		}
	} else {
		m_iVoxelSuperSampling = pCudaProjector->getVoxelSuperSampling();
		m_GPUIndices = pCudaProjector->getGPUIndices();
	}

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaBackProjectionAlgorithm3D::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaBackProjectionAlgorithm3D", this, _cfg);

	// initialization of parent class
	if (!CReconstructionAlgorithm3D::initialize(_cfg)) {
		return false;
	}

	initializeFromProjector();

	bool ok = true;

	// Deprecated options
	ok &= CR.getOptionInt("VoxelSuperSampling", m_iVoxelSuperSampling, m_iVoxelSuperSampling);
	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionIntArray("GPUIndex", m_GPUIndices, true);
	else if (CR.hasOption("GPUindex"))
		ok &= CR.getOptionIntArray("GPUindex", m_GPUIndices, true);

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
	assert(!m_bIsInitialized);

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
// SIRT-weighted backprojection
// This computes the column weights, divides by them, and adds the
// result to the current volume. This is both more expensive and more
// GPU memory intensive than the regular BP, but allows saving system RAM.

static bool astraCudaBP_SIRTWeighted(CFloat32VolumeData3D *pVolume,
                                     const CFloat32ProjectionData3D* pProjections,
                                     int iGPUIndex, int iVoxelSuperSampling)
{
	astraCUDA3d::SProjectorParams3D params;

	params.iRaysPerVoxelDim = iVoxelSuperSampling;

	Geometry3DParameters projs = astra::convertAstraGeometry(&pVolume->getGeometry(), &pProjections->getGeometry());
	params.volScale = projs.getVolScale();

	if (!projs.isValid())
		return false;

	if (iGPUIndex != -1) {
		if (!astraCUDA3d::setGPUIndex(iGPUIndex))
			return false;
	}

	std::array<int, 3> volDims = pVolume->getShape();
	std::array<int, 3> projDims = pProjections->getShape();

	CDataStorage *s;
	s = astraCUDA3d::allocateGPUMemory(volDims[0], volDims[1], volDims[2], astraCUDA3d::INIT_ZERO);
	if (!s) {
		return false;
	}
	CData3D *D_volData = new CData3D(volDims[0], volDims[1], volDims[2], s);

	s = astraCUDA3d::allocateGPUMemory(volDims[0], volDims[1], volDims[2], astraCUDA3d::INIT_ZERO);
	if (!s) {
		astraCUDA3d::freeGPUMemory(D_volData);
		delete D_volData;
		return false;
	}
	CData3D *D_pixelWeight = new CData3D(volDims[0], volDims[1], volDims[2], s);

	s = astraCUDA3d::allocateGPUMemory(projDims[0], projDims[1], projDims[2], astraCUDA3d::INIT_NO);
	if (!s) {
		astraCUDA3d::freeGPUMemory(D_volData);
		astraCUDA3d::freeGPUMemory(D_pixelWeight);
		delete D_volData;
		delete D_pixelWeight;
		return false;
	}
	CData3D *D_projData = new CData3D(projDims[0], projDims[1], projDims[2], s);

	// Compute weights
	bool ok = true;
	ok &= astraCUDA3d::processVol3D<astraCUDA3d::opSet>(D_projData, 1.0f);

	ok &= astraCUDA3d::BP(D_pixelWeight, D_projData, projs, params);

	ok &= astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(D_pixelWeight);
	if (!ok) {
		astraCUDA3d::freeGPUMemory(D_volData);
		astraCUDA3d::freeGPUMemory(D_pixelWeight);
		astraCUDA3d::freeGPUMemory(D_projData);
		delete D_volData;
		delete D_pixelWeight;
		delete D_projData;
		return false;
	}

	ok &= astraCUDA3d::copyToGPUMemory(pProjections, D_projData);
	ok &= astraCUDA3d::BP(D_volData, D_projData, projs, params);

	// Multiply with weights
	ok &= astraCUDA3d::processVol3D<astraCUDA3d::opMul>(D_volData, D_pixelWeight);

	// Upload previous iterate to D_pixelWeight...
	ok &= astraCUDA3d::copyToGPUMemory(pVolume, D_pixelWeight);
	if (!ok) {
		astraCUDA3d::freeGPUMemory(D_volData);
		astraCUDA3d::freeGPUMemory(D_pixelWeight);
		astraCUDA3d::freeGPUMemory(D_projData);
		delete D_volData;
		delete D_pixelWeight;
		delete D_projData;
		return false;
	}
	// ...and add it to the weighted BP
	ok &= astraCUDA3d::processVol3D<astraCUDA3d::opAdd>(D_volData, D_pixelWeight);

	// Then copy the result back
	ok &= astraCUDA3d::copyFromGPUMemory(pVolume, D_volData);

	astraCUDA3d::freeGPUMemory(D_volData);
	astraCUDA3d::freeGPUMemory(D_pixelWeight);
	astraCUDA3d::freeGPUMemory(D_projData);

	delete D_volData;
	delete D_pixelWeight;
	delete D_projData;

	return ok;
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

	if (m_bSIRTWeighting) {
		ASTRA_ASSERT(m_pSinogram->isFloat32Memory());
		ASTRA_ASSERT(m_pReconstruction->isFloat32Memory());

		// TODO: Warn if multiple GPUs specified but only one is used?
		int iGPUIndex = -1;
		if (!m_GPUIndices.empty())
			iGPUIndex = m_GPUIndices[0];

		return astraCudaBP_SIRTWeighted(m_pReconstruction,
		                                m_pSinogram,
		                                iGPUIndex, m_iVoxelSuperSampling);
	} else {
		CCompositeGeometryManager cgm;
		if (!m_GPUIndices.empty())
			cgm.setGPUIndices(m_GPUIndices);

		return cgm.doBP(m_pProjector, pReconMem, pSinoMem);
	}

}


} // namespace astra

#endif // ASTRA_CUDA
