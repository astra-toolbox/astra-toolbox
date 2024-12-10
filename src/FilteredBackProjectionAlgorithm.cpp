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

#include "astra/FilteredBackProjectionAlgorithm.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#include "astra/AstraObjectManager.h"
#include "astra/ParallelBeamLineKernelProjector2D.h"
#include "astra/Fourier.h"
#include "astra/DataProjector.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

const int FFT = 1;
const int IFFT = -1;

//----------------------------------------------------------------------------------------
// Constructor
CFilteredBackProjectionAlgorithm::CFilteredBackProjectionAlgorithm() 
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CFilteredBackProjectionAlgorithm::~CFilteredBackProjectionAlgorithm() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CFilteredBackProjectionAlgorithm::_clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CFilteredBackProjectionAlgorithm::clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bIsInitialized = false;

	m_filterConfig.m_pfCustomFilter.clear();
}


//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CFilteredBackProjectionAlgorithm::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("FilteredBackProjectionAlgorithm", this, _cfg);

	bool ok = true;
	int id = -1;

	ok &= CR.getRequiredID("ProjectorId", id);
	m_pProjector = CProjector2DManager::getSingleton().get(id);

	ok &= CR.getRequiredID("ProjectionDataId", id);
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));

	ok &= CR.getRequiredID("ReconstructionDataId", id);
	m_pReconstruction = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));

	if (CR.has("ProjectionIndex")) {
		ASTRA_ERROR("ProjectionIndex is no longer available. Manually adjust the sinogram instead.");
		return false;
	}

	m_filterConfig = getFilterConfigForAlgorithm(_cfg, this);

	const CParallelProjectionGeometry2D* parprojgeom = dynamic_cast<const CParallelProjectionGeometry2D*>(&m_pSinogram->getGeometry());
	if (!parprojgeom) {
		ASTRA_ERROR("FBP currently only supports parallel projection geometries.");
		return false;
	}

	// TODO: check that the angles are linearly spaced between 0 and pi

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Initialize
bool CFilteredBackProjectionAlgorithm::initialize(CProjector2D* _pProjector, 
												  CFloat32VolumeData2D* _pVolume,
												  CFloat32ProjectionData2D* _pSinogram)
{
	clear();

	// store classes
	m_pProjector = _pProjector;
	m_pReconstruction = _pVolume;
	m_pSinogram = _pSinogram;

	m_filterConfig = SFilterConfig();

	// TODO: check that the angles are linearly spaced between 0 and pi

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Check
bool CFilteredBackProjectionAlgorithm::_check() 
{
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm2D::_check(), "FBP", "Error in ReconstructionAlgorithm2D initialization");

	ASTRA_CONFIG_CHECK(m_filterConfig.m_eType != FILTER_ERROR, "FBP", "Invalid filter name.");

	if((m_filterConfig.m_eType == FILTER_PROJECTION) || (m_filterConfig.m_eType == FILTER_SINOGRAM) || (m_filterConfig.m_eType == FILTER_RPROJECTION) || (m_filterConfig.m_eType == FILTER_RSINOGRAM))
	{
		ASTRA_CONFIG_CHECK(!m_filterConfig.m_pfCustomFilter.empty(), "FBP", "Invalid filter pointer.");
	}

	ASTRA_CONFIG_CHECK(checkCustomFilterSize(m_filterConfig, m_pSinogram->getGeometry()), "FBP", "Filter size mismatch");

	// success
	return true;
}


//----------------------------------------------------------------------------------------
// Iterate
bool CFilteredBackProjectionAlgorithm::run(int _iNrIterations)
{
	ASTRA_ASSERT(m_bIsInitialized);

	// Filter sinogram
	CFloat32ProjectionData2D filteredSinogram(m_pSinogram->getGeometry(), m_pSinogram->getData());
	performFiltering(&filteredSinogram);

	// Back project
	m_pReconstruction->setData(0.0f);
	projectData(m_pProjector,
	            DefaultBPPolicy(m_pReconstruction, &filteredSinogram));

	// Scale data
	const CVolumeGeometry2D& volGeom = m_pProjector->getVolumeGeometry();
	const CProjectionGeometry2D& projGeom = m_pProjector->getProjectionGeometry();

	int iAngleCount = projGeom.getProjectionAngleCount();
	float fPixelArea = volGeom.getPixelArea();
	(*m_pReconstruction) *= PI/(2*iAngleCount*fPixelArea);

	m_pReconstruction->updateStatistics();

	return true;
}


//----------------------------------------------------------------------------------------
void CFilteredBackProjectionAlgorithm::performFiltering(CFloat32ProjectionData2D * _pFilteredSinogram)
{
	ASTRA_ASSERT(_pFilteredSinogram != NULL);
	ASTRA_ASSERT(_pFilteredSinogram->getAngleCount() == m_pSinogram->getAngleCount());
	ASTRA_ASSERT(_pFilteredSinogram->getDetectorCount() == m_pSinogram->getDetectorCount());

	ASTRA_ASSERT(m_filterConfig.m_eType != FILTER_ERROR);
	if (m_filterConfig.m_eType == FILTER_NONE)
		return;

	int iAngleCount = m_pProjector->getProjectionGeometry().getProjectionAngleCount();
	int iDetectorCount = m_pProjector->getProjectionGeometry().getDetectorCount();


	int zpDetector = calcNextPowerOfTwo(2 * m_pSinogram->getDetectorCount());
	int iHalfFFTSize = astra::calcFFTFourierSize(zpDetector);

	// cdft setup
	int *ip = new int[int(2+sqrt((float)zpDetector)+1)];
	ip[0] = 0;
	float32 *w = new float32[zpDetector/2];

	// Create filter
	bool bFilterMultiAngle = false;
	bool bFilterComplex = false;
	float *pfFilter = 0;
	float *pfFilter_delete = 0;
	switch (m_filterConfig.m_eType) {
		case FILTER_ERROR:
		case FILTER_NONE:
			// Should have been handled before
			ASTRA_ASSERT(false);
			return;
		case FILTER_PROJECTION:
			// Fourier space, real, half the coefficients (because symmetric)
			// 1 x iHalfFFTSize
			pfFilter = &m_filterConfig.m_pfCustomFilter[0];
			break;
		case FILTER_SINOGRAM:
			bFilterMultiAngle = true;
			pfFilter = &m_filterConfig.m_pfCustomFilter[0];
			break;
		case FILTER_RSINOGRAM:
			bFilterMultiAngle = true;
			// fall-through
		case FILTER_RPROJECTION:
		{
			bFilterComplex = true;

			int count = bFilterMultiAngle ? iAngleCount : 1;
			// Spatial, real, full convolution kernel
			// Center in center (or right-of-center for even sized.)
			// I.e., 0 1 0 and 0 0 1 0 both correspond to the identity

			pfFilter = new float[2 * zpDetector * count];
			pfFilter_delete = pfFilter;

			int iUsedFilterWidth = min(m_filterConfig.m_iCustomFilterWidth, zpDetector);
			int iStartFilterIndex = (m_filterConfig.m_iCustomFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = m_filterConfig.m_iCustomFilterWidth / 2;

			for (int i = 0; i < count; ++i) {
				float *rOut = pfFilter + i * 2 * zpDetector;
				float *rIn = &m_filterConfig.m_pfCustomFilter[i * m_filterConfig.m_iCustomFilterWidth];
				memset(rOut, 0, sizeof(float) * 2 * zpDetector);

				for(int j = iStartFilterIndex; j < iMaxFilterIndex; j++) {
					int iFFTInFilterIndex = (j + zpDetector - iFilterShiftSize) % zpDetector;
					rOut[2 * iFFTInFilterIndex] = rIn[j];
				}

				cdft(2*zpDetector, -1, rOut, ip, w);
			}

			break;
		}
		default:
			pfFilter = genFilter(m_filterConfig, zpDetector, iHalfFFTSize);
			pfFilter_delete = pfFilter;
	}

	float32* pf = new float32[2 * iAngleCount * zpDetector];

	// Copy and zero-pad data
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		float32* pfDataRow = _pFilteredSinogram->getData() + iAngle * iDetectorCount;
		for (int iDetector = 0; iDetector < iDetectorCount; ++iDetector) {
			pfRow[2*iDetector] = pfDataRow[iDetector];
			pfRow[2*iDetector+1] = 0.0f;
		}
		for (int iDetector = iDetectorCount; iDetector < zpDetector; ++iDetector) {
			pfRow[2*iDetector] = 0.0f;
			pfRow[2*iDetector+1] = 0.0f;
		}
	}

	// in-place FFT
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		cdft(2*zpDetector, -1, pfRow, ip, w);
	}

	// Filter
	if (bFilterComplex) {
		for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
			float32* pfRow = pf + iAngle * 2 * zpDetector;
			float *pfFilterRow = pfFilter;
			if (bFilterMultiAngle)
				pfFilterRow += iAngle * 2 * zpDetector;

			for (int i = 0; i < zpDetector; ++i) {
				float re = pfRow[2*i] * pfFilterRow[2*i] - pfRow[2*i+1] * pfFilterRow[2*i+1];
				float im = pfRow[2*i] * pfFilterRow[2*i+1] + pfRow[2*i+1] * pfFilterRow[2*i];
				pfRow[2*i] = re;
				pfRow[2*i+1] = im;
			}
		}
	} else {
		for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
			float32* pfRow = pf + iAngle * 2 * zpDetector;
			float *pfFilterRow = pfFilter;
			if (bFilterMultiAngle)
				pfFilterRow += iAngle * iHalfFFTSize;
			for (int iDetector = 0; iDetector < iHalfFFTSize; ++iDetector) {
				pfRow[2*iDetector] *= pfFilterRow[iDetector];
				pfRow[2*iDetector+1] *= pfFilterRow[iDetector];
			}
			for (int iDetector = iHalfFFTSize; iDetector < zpDetector; ++iDetector) {
				pfRow[2*iDetector] *= pfFilterRow[zpDetector - iDetector];
				pfRow[2*iDetector+1] *= pfFilterRow[zpDetector - iDetector];
			}
		}
	}

	// in-place inverse FFT
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		cdft(2*zpDetector, 1, pfRow, ip, w);
	}

	// Copy data back
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		float32* pfDataRow = _pFilteredSinogram->getData() + iAngle * iDetectorCount;
		for (int iDetector = 0; iDetector < iDetectorCount; ++iDetector)
			pfDataRow[iDetector] = pfRow[2*iDetector] / zpDetector;
	}

	delete[] pf;
	delete[] w;
	delete[] ip;
	delete[] pfFilter_delete;
}

}
