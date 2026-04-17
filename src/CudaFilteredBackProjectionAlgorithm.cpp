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

#include <astra/CudaFilteredBackProjectionAlgorithm.h>
#include <astra/FanFlatProjectionGeometry2D.h>

#include "astra/AstraObjectManager.h"
#include "astra/CudaProjector2D.h"
#include "astra/Filters.h"
#include "astra/cuda/2d/fbp.h"
#include "astra/cuda/2d/mem2d.h"

#include "astra/Logging.h"

#include <cstring>

using namespace std;
using namespace astra;

CCudaFilteredBackProjectionAlgorithm::CCudaFilteredBackProjectionAlgorithm()
	: m_filterConfig(), m_bShortScan(false), m_filter(nullptr)
{

}

CCudaFilteredBackProjectionAlgorithm::~CCudaFilteredBackProjectionAlgorithm()
{
	if (m_filter)
		astraCUDA::freeFilter(m_filter);

}

bool CCudaFilteredBackProjectionAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaFilteredBackProjectionAlgorithm", this, _cfg);

	if (!CCudaReconstructionAlgorithm2D::initialize(_cfg))
		return false;

	m_filterConfig = getFilterConfigForAlgorithm(_cfg, this);

	// Fan beam short scan mode
	m_bShortScan = false;
	if (m_pSinogram && (dynamic_cast<const CFanFlatProjectionGeometry2D*>(&m_pSinogram->getGeometry())
			|| dynamic_cast<const CFanFlatVecProjectionGeometry2D*>(&m_pSinogram->getGeometry()))) {
		bool ok = true;
		ok &= CR.getOptionBool("ShortScan", m_bShortScan, false);
		if (!ok)
			return false;
	}

	initializeFromProjector();

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);
	m_filter = astraCUDA::prepareFilter(m_filterConfig, m_geometry.getDims());

	m_bIsInitialized = check();
	return m_bIsInitialized;
}

bool CCudaFilteredBackProjectionAlgorithm::initialize(CFloat32ProjectionData2D * _pSinogram, CFloat32VolumeData2D * _pReconstruction, E_FBPFILTER _eFilter, const float * _pfFilter /* = NULL */, int _iFilterWidth /* = 0 */, int _iGPUIndex /* = 0 */, float _fFilterParameter /* = -1.0f */)
{
	assert(!m_bIsInitialized);

	// required classes
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;
	m_iGPUIndex = _iGPUIndex;

	m_filterConfig.m_eType = _eFilter;
	m_filterConfig.m_iCustomFilterWidth = _iFilterWidth;
	m_bShortScan = false;

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

		m_filterConfig.m_pfCustomFilter.resize(iFilterElementCount);
		memcpy(&m_filterConfig.m_pfCustomFilter[0], _pfFilter, iFilterElementCount * sizeof(float));
	}
	else
	{
		m_filterConfig.m_pfCustomFilter.clear();
	}

	m_filterConfig.m_fParameter = _fFilterParameter;

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);
	m_filter = astraCUDA::prepareFilter(m_filterConfig, m_geometry.getDims());

	m_bIsInitialized = check();
	return m_bIsInitialized;
}

bool CCudaFilteredBackProjectionAlgorithm::check()
{
	// check pointers
	ASTRA_CONFIG_CHECK(m_pSinogram, "FBP_CUDA", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pReconstruction, "FBP_CUDA", "Invalid Reconstruction Data Object.");

	ASTRA_CONFIG_CHECK(m_filterConfig.m_eType != FILTER_ERROR, "FBP_CUDA", "Invalid filter name.");

	if((m_filterConfig.m_eType == FILTER_PROJECTION) || (m_filterConfig.m_eType == FILTER_SINOGRAM) || (m_filterConfig.m_eType == FILTER_RPROJECTION) || (m_filterConfig.m_eType == FILTER_RSINOGRAM))
	{
		ASTRA_CONFIG_CHECK(!m_filterConfig.m_pfCustomFilter.empty(), "FBP_CUDA", "Invalid filter pointer.");
	}

	// check initializations
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "FBP_CUDA", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isInitialized(), "FBP_CUDA", "Reconstruction Data Object Not Initialized.");

	// check gpu index
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FBP_CUDA", "GPUIndex must be a non-negative integer or -1.");
	// check pixel supersampling
	ASTRA_CONFIG_CHECK(m_params.iRaysPerPixelDim >= 0, "FBP_CUDA", "PixelSuperSampling must be a non-negative integer.");

	ASTRA_CONFIG_CHECK(checkCustomFilterSize(m_filterConfig, m_pSinogram->getGeometry()), "FBP_CUDA", "Filter size mismatch");


	// success
	m_bIsInitialized = true;
	return true;
}

bool CCudaFilteredBackProjectionAlgorithm::run(int /*_iNrIterations*/)
{
	assert(m_bIsInitialized);

	bool ok = true;

	std::array<int, 2> volDims = m_pReconstruction->getShape();
	std::array<int, 2> projDims = m_pSinogram->getShape();

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	CDataStorage *s;
	s = astraCUDA::allocateGPUMemory(volDims[0], volDims[1], astraCUDA::INIT_NO);
	if (!s) {
		return false;
	}
	CData2D *D_volData = new CData2D(volDims[0], volDims[1], s);

	s = astraCUDA::allocateGPUMemory(projDims[0], projDims[1], astraCUDA::INIT_NO);
	if (!s) {
		astraCUDA::freeGPUMemory(D_volData);
		delete D_volData;
		return false;
	}
	CData2D *D_projData = new CData2D(projDims[0], projDims[1], s);

	if (m_pSinogram->isFloat32Memory()) {
		ok &= astraCUDA::copyToGPUMemory(m_pSinogram, D_projData);
	} else if (m_pSinogram->isFloat32GPU()) {
		// TODO: re-use memory instead of copying
		// (need to ensure everything works when pitches are not consistent)
		ok &= astraCUDA::assignGPUMemory(D_projData, m_pSinogram);
	} else {
		ok = false;
	}

	astraCUDA::SProjectorParams2D params = m_params;
	float fPixelArea = m_pReconstruction->getGeometry().getPixelArea();
	params.fOutputScale *= 1.0f / fPixelArea;

	if (ok)
		ok &= FBP(D_volData, D_projData, m_geometry, params, m_filter, m_bShortScan);

	if (ok) {
		if (m_pReconstruction->isFloat32Memory()) {
			ok &= astraCUDA::copyFromGPUMemory(m_pReconstruction, D_volData);
		} else if (m_pReconstruction->isFloat32GPU()) {
			// TODO: re-use memory instead of copying
			// (need to ensure everything works when pitches are not consistent)
			ok &= astraCUDA::assignGPUMemory(m_pReconstruction, D_volData);
		} else {
			ok = false;
		}
	}


	astraCUDA::freeGPUMemory(D_volData);
	astraCUDA::freeGPUMemory(D_projData);
	delete D_volData;
	delete D_projData;

	return ok;
}
