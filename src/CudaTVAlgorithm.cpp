/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#ifdef ASTRA_CUDA

#include "astra/CudaTVAlgorithm.h"

#include "astra/AstraObjectManager.h"

#include "../cuda/2d/tv.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaTVAlgorithm::type = "TV_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaTVAlgorithm::CCudaTVAlgorithm()
{
	m_bIsInitialized = false;
	CCudaReconstructionAlgorithm2D::_clear();

	m_pMinMask = 0;
	m_pMaxMask = 0;

	m_fLambda = 1.0f;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaTVAlgorithm::~CCudaTVAlgorithm()
{
	// The actual work is done by ~CCudaReconstructionAlgorithm2D

	m_pMinMask = 0;
	m_pMaxMask = 0;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaTVAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaTVAlgorithm", this, _cfg);

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

	m_fLambda = _cfg.self.getOptionNumerical("Regularization", 1.0f);
	CC.markOptionParsed("Regularization");

	m_pAlgo = new astraCUDA::TV();
	m_bAlgoInit = false;

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaTVAlgorithm::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram,
                                     CFloat32VolumeData2D* _pReconstruction)
{
	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction);

	if (!m_bIsInitialized)
		return false;



	m_pAlgo = new astraCUDA::TV();
	m_bAlgoInit = false;
	m_fLambda = 1.0f;

	return true;
}

//----------------------------------------------------------------------------------------

void CCudaTVAlgorithm::initCUDAAlgorithm()
{
	CCudaReconstructionAlgorithm2D::initCUDAAlgorithm();

	astraCUDA::TV* pTV = dynamic_cast<astraCUDA::TV*>(m_pAlgo);

	if (m_pMinMask || m_pMaxMask) {
		const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();
		const float *pfMinMaskData = 0;
		const float *pfMaxMaskData = 0;
		if (m_pMinMask) pfMinMaskData = m_pMinMask->getDataConst();
		if (m_pMaxMask) pfMaxMaskData = m_pMaxMask->getDataConst();
		bool ok = pTV->uploadMinMaxMasks(pfMinMaskData, pfMaxMaskData, volgeom.getGridColCount());
		ASTRA_ASSERT(ok);
	}

	pTV->setRegularization(m_fLambda);
}


} // namespace astra

#endif // ASTRA_CUDA
