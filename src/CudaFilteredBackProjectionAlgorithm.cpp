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

#include <astra/CudaFilteredBackProjectionAlgorithm.h>
#include <astra/FanFlatProjectionGeometry2D.h>

#include "astra/AstraObjectManager.h"
#include "astra/CudaProjector2D.h"
#include "astra/Filters.h"
#include "astra/cuda/2d/astra.h"
#include "astra/cuda/2d/fbp.h"

#include "astra/Logging.h"

using namespace std;
using namespace astra;

string CCudaFilteredBackProjectionAlgorithm::type = "FBP_CUDA";

CCudaFilteredBackProjectionAlgorithm::CCudaFilteredBackProjectionAlgorithm()
{
	m_bIsInitialized = false;
	CCudaReconstructionAlgorithm2D::_clear();
}

CCudaFilteredBackProjectionAlgorithm::~CCudaFilteredBackProjectionAlgorithm()
{
	delete[] m_filterConfig.m_pfCustomFilter;
	m_filterConfig.m_pfCustomFilter = NULL;
}

bool CCudaFilteredBackProjectionAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaFilteredBackProjectionAlgorithm", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized)
	{
		clear();
	}

	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_cfg);
	if (!m_bIsInitialized)
		return false;

	m_filterConfig = getFilterConfigForAlgorithm(_cfg, this);

	// Fan beam short scan mode
	if (m_pSinogram && dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pSinogram->getGeometry())) {
		m_bShortScan = (int)_cfg.self.getOptionBool("ShortScan", false);
		CC.markOptionParsed("ShortScan");
	}

	initializeFromProjector();


	m_pAlgo = new astraCUDA::FBP();
	m_bAlgoInit = false;

	return check();
}

bool CCudaFilteredBackProjectionAlgorithm::initialize(CFloat32ProjectionData2D * _pSinogram, CFloat32VolumeData2D * _pReconstruction, E_FBPFILTER _eFilter, const float * _pfFilter /* = NULL */, int _iFilterWidth /* = 0 */, int _iGPUIndex /* = 0 */, float _fFilterParameter /* = -1.0f */)
{
	// if already initialized, clear first
	if (m_bIsInitialized)
	{
		clear();
	}

	// required classes
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;
	m_iGPUIndex = _iGPUIndex;

	m_filterConfig.m_eType = _eFilter;
	m_filterConfig.m_iCustomFilterWidth = _iFilterWidth;
	m_bShortScan = false;

	// success
	m_bIsInitialized = true;

	m_pAlgo = new astraCUDA::FBP();
	m_bAlgoInit = false;

	if(_pfFilter != NULL)
	{
		int iFilterElementCount = 0;

		if((m_filterConfig.m_eType != FILTER_SINOGRAM) && (m_filterConfig.m_eType != FILTER_RSINOGRAM))
		{
			iFilterElementCount = _iFilterWidth;
		}
		else
		{
			iFilterElementCount = m_pSinogram->getAngleCount();
		}

		m_filterConfig.m_pfCustomFilter = new float[iFilterElementCount];
		memcpy(m_filterConfig.m_pfCustomFilter, _pfFilter, iFilterElementCount * sizeof(float));
	}
	else
	{
		m_filterConfig.m_pfCustomFilter = NULL;
	}

	m_filterConfig.m_fParameter = _fFilterParameter;

	return check();
}


void CCudaFilteredBackProjectionAlgorithm::initCUDAAlgorithm()
{
	CCudaReconstructionAlgorithm2D::initCUDAAlgorithm();

	astraCUDA::FBP* pFBP = dynamic_cast<astraCUDA::FBP*>(m_pAlgo);

	bool ok = pFBP->setFilter(m_filterConfig);
	if (!ok) {
		ASTRA_ERROR("CCudaFilteredBackProjectionAlgorithm: Failed to set filter");
		ASTRA_ASSERT(ok);
	}

	ok &= pFBP->setShortScan(m_bShortScan);
	if (!ok) {
		ASTRA_ERROR("CCudaFilteredBackProjectionAlgorithm: Failed to set short-scan mode");
	}
}


bool CCudaFilteredBackProjectionAlgorithm::check()
{
	// check pointers
	ASTRA_CONFIG_CHECK(m_pSinogram, "FBP_CUDA", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pReconstruction, "FBP_CUDA", "Invalid Reconstruction Data Object.");

	ASTRA_CONFIG_CHECK(m_filterConfig.m_eType != FILTER_ERROR, "FBP_CUDA", "Invalid filter name.");

	if((m_filterConfig.m_eType == FILTER_PROJECTION) || (m_filterConfig.m_eType == FILTER_SINOGRAM) || (m_filterConfig.m_eType == FILTER_RPROJECTION) || (m_filterConfig.m_eType == FILTER_RSINOGRAM))
	{
		ASTRA_CONFIG_CHECK(m_filterConfig.m_pfCustomFilter, "FBP_CUDA", "Invalid filter pointer.");
	}

	// check initializations
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "FBP_CUDA", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isInitialized(), "FBP_CUDA", "Reconstruction Data Object Not Initialized.");

	// check gpu index
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FBP_CUDA", "GPUIndex must be a non-negative integer or -1.");
	// check pixel supersampling
	ASTRA_CONFIG_CHECK(m_iPixelSuperSampling >= 0, "FBP_CUDA", "PixelSuperSampling must be a non-negative integer.");

	ASTRA_CONFIG_CHECK(checkCustomFilterSize(m_filterConfig, *m_pSinogram->getGeometry()), "FBP_CUDA", "Filter size mismatch");


	// success
	m_bIsInitialized = true;
	return true;
}


