/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#include "astra/CudaSirtAlgorithm.h"

#include "astra/AstraObjectManager.h"

#include "astra/cuda/2d/sirt.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaSirtAlgorithm::type = "SIRT_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaSirtAlgorithm::CCudaSirtAlgorithm() 
{
	m_bIsInitialized = false;
	CCudaReconstructionAlgorithm2D::_clear();

	m_pMinMask = 0;
	m_pMaxMask = 0;

	m_fLambda = 1.0f;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSirtAlgorithm::~CCudaSirtAlgorithm() 
{
	// The actual work is done by ~CCudaReconstructionAlgorithm2D

	m_pMinMask = 0;
	m_pMaxMask = 0;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSirtAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaSirtAlgorithm", this, _cfg);

	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_cfg);

	if (!m_bIsInitialized)
		return false;

	// min/max masks
	if (_cfg.self.hasOption("MinMaskId")) {
		int id = _cfg.self.getOptionInt("MinMaskId");
		m_pMinMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("MinMaskId");
	if (_cfg.self.hasOption("MaxMaskId")) {
		int id = _cfg.self.getOptionInt("MaxMaskId");
		m_pMaxMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("MaxMaskId");

	m_fLambda = _cfg.self.getOptionNumerical("Relaxation", 1.0f);
	CC.markOptionParsed("Relaxation");

	m_pAlgo = new astraCUDA::SIRT();
	m_bAlgoInit = false;

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaSirtAlgorithm::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram, 
                                     CFloat32VolumeData2D* _pReconstruction)
{
	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction);

	if (!m_bIsInitialized)
		return false;

	

	m_pAlgo = new astraCUDA::SIRT();
	m_bAlgoInit = false;
	m_fLambda = 1.0f;

	return true;
}

//---------------------------------------------------------------------------------------
void CCudaSirtAlgorithm::updateSlice(CFloat32ProjectionData2D* _pSinogram,
                                     CFloat32VolumeData2D* _pReconstruction)
{
    m_pSinogram = _pSinogram;
    m_pReconstruction = _pReconstruction;
}

//----------------------------------------------------------------------------------------

void CCudaSirtAlgorithm::initCUDAAlgorithm()
{
	CCudaReconstructionAlgorithm2D::initCUDAAlgorithm();

	astraCUDA::SIRT* pSirt = dynamic_cast<astraCUDA::SIRT*>(m_pAlgo);

	if (m_pMinMask || m_pMaxMask) {
		const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();
		const float *pfMinMaskData = 0;
		const float *pfMaxMaskData = 0;
		if (m_pMinMask) pfMinMaskData = m_pMinMask->getDataConst();
		if (m_pMaxMask) pfMaxMaskData = m_pMaxMask->getDataConst();
		bool ok = pSirt->uploadMinMaxMasks(pfMinMaskData, pfMaxMaskData, volgeom.getGridColCount());
		ASTRA_ASSERT(ok);
	}

	pSirt->setRelaxation(m_fLambda);
}


} // namespace astra

#endif // ASTRA_CUDA
