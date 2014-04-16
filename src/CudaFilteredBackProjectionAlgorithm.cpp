/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

#include <astra/CudaFilteredBackProjectionAlgorithm.h>
#include <astra/FanFlatProjectionGeometry2D.h>
#include <boost/lexical_cast.hpp>
#include <cstring>

#include "astra/AstraObjectManager.h"

#include <astra/Logger.h>

using namespace std;
using namespace astra;

string CCudaFilteredBackProjectionAlgorithm::type = "FBP_CUDA";

CCudaFilteredBackProjectionAlgorithm::CCudaFilteredBackProjectionAlgorithm()
{
	m_bIsInitialized = false;
	CReconstructionAlgorithm2D::_clear();
	m_pFBP = 0;
	m_pfFilter = NULL;
	m_fFilterParameter = -1.0f;
	m_fFilterD = 1.0f;
}

CCudaFilteredBackProjectionAlgorithm::~CCudaFilteredBackProjectionAlgorithm()
{
	if(m_pfFilter != NULL)
	{
		delete [] m_pfFilter;
		m_pfFilter = NULL;
	}

	if(m_pFBP != NULL)
	{
		delete m_pFBP;
		m_pFBP = NULL;
	}
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

	// sinogram data
	XMLNode* node = _cfg.self->getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "CudaFBP", "No ProjectionDataId tag specified.");
	int id = boost::lexical_cast<int>(node->getContent());
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	ASTRA_DELETE(node);
	CC.markNodeParsed("ProjectionDataId");

	// reconstruction data
	node = _cfg.self->getSingleNode("ReconstructionDataId");
	ASTRA_CONFIG_CHECK(node, "CudaFBP", "No ReconstructionDataId tag specified.");
	id = boost::lexical_cast<int>(node->getContent());
	m_pReconstruction = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	ASTRA_DELETE(node);
	CC.markNodeParsed("ReconstructionDataId");

	// filter type
	node = _cfg.self->getSingleNode("FilterType");
	if(node != NULL)
	{
		m_eFilter = _convertStringToFilter(node->getContent().c_str());
	}
	else
	{
		m_eFilter = FILTER_RAMLAK;
	}
	CC.markNodeParsed("FilterType");
	ASTRA_DELETE(node);
	
	// filter
	node = _cfg.self->getSingleNode("FilterSinogramId");
	if(node != NULL)
	{
		int id = boost::lexical_cast<int>(node->getContent());
		const CFloat32ProjectionData2D * pFilterData = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
		m_iFilterWidth = pFilterData->getGeometry()->getDetectorCount();
		int iFilterProjectionCount = pFilterData->getGeometry()->getProjectionAngleCount();

		m_pfFilter = new float[m_iFilterWidth * iFilterProjectionCount];
		memcpy(m_pfFilter, pFilterData->getDataConst(), sizeof(float) * m_iFilterWidth * iFilterProjectionCount);
	}
	else
	{
		m_iFilterWidth = 0;
		m_pfFilter = NULL;
	}
	CC.markNodeParsed("FilterSinogramId"); // TODO: Only for some types!
	ASTRA_DELETE(node);

	// filter parameter
	node = _cfg.self->getSingleNode("FilterParameter");
	if(node != NULL)
	{
		float fParameter = boost::lexical_cast<float>(node->getContent());
		m_fFilterParameter = fParameter;
	}
	else
	{
		m_fFilterParameter = -1.0f;
	}
	CC.markNodeParsed("FilterParameter"); // TODO: Only for some types!
	ASTRA_DELETE(node);

	// D value
	node = _cfg.self->getSingleNode("FilterD");
	if(node != NULL)
	{
		float fD = boost::lexical_cast<float>(node->getContent());
		m_fFilterD = fD;
	}
	else
	{
		m_fFilterD = 1.0f;
	}
	CC.markNodeParsed("FilterD"); // TODO: Only for some types!
	ASTRA_DELETE(node);

	// GPU number
	m_iGPUIndex = (int)_cfg.self->getOptionNumerical("GPUindex", -1);
	CC.markOptionParsed("GPUindex");

	// Pixel supersampling factor
	m_iPixelSuperSampling = (int)_cfg.self->getOptionNumerical("PixelSuperSampling", 1);
	CC.markOptionParsed("PixelSuperSampling");


	m_pFBP = new AstraFBP;
	m_bAstraFBPInit = false;

	// success
	m_bIsInitialized = true;
	return m_bIsInitialized;
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

	m_eFilter = _eFilter;
	m_iFilterWidth = _iFilterWidth;

	// success
	m_bIsInitialized = true;

	m_pFBP = new AstraFBP;

	m_bAstraFBPInit = false;

	if(_pfFilter != NULL)
	{
		int iFilterElementCount = 0;

		if((_eFilter != FILTER_SINOGRAM) && (_eFilter != FILTER_RSINOGRAM))
		{
			iFilterElementCount = _iFilterWidth;
		}
		else
		{
			iFilterElementCount = m_pSinogram->getAngleCount();
		}

		m_pfFilter = new float[iFilterElementCount];
		memcpy(m_pfFilter, _pfFilter, iFilterElementCount * sizeof(float));
	}
	else
	{
		m_pfFilter = NULL;
	}

	m_fFilterParameter = _fFilterParameter;

	return m_bIsInitialized;
}

void CCudaFilteredBackProjectionAlgorithm::run(int _iNrIterations /* = 0 */)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	if (!m_bAstraFBPInit) {

		const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();
		const CParallelProjectionGeometry2D* parprojgeom = dynamic_cast<CParallelProjectionGeometry2D*>(m_pSinogram->getGeometry());
		const CFanFlatProjectionGeometry2D* fanprojgeom = dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pSinogram->getGeometry());

		bool ok = true;

		// TODO: off-center geometry, non-square pixels
		ok &= m_pFBP->setReconstructionGeometry(volgeom.getGridColCount(),
		                                         volgeom.getGridRowCount(),
		                                         volgeom.getPixelLengthX());
		// TODO: off-center geometry
		int iDetectorCount;
		if (parprojgeom) {
			ok &= m_pFBP->setProjectionGeometry(parprojgeom->getProjectionAngleCount(),
			                                     parprojgeom->getDetectorCount(),
			                                     parprojgeom->getProjectionAngles(),
			                                     parprojgeom->getDetectorWidth());
			iDetectorCount = parprojgeom->getDetectorCount();
		} else if (fanprojgeom) {
			ok &= m_pFBP->setFanGeometry(fanprojgeom->getProjectionAngleCount(),
			                                     fanprojgeom->getDetectorCount(),
			                                     fanprojgeom->getProjectionAngles(),
			                                     fanprojgeom->getOriginSourceDistance(),
			                                     fanprojgeom->getOriginDetectorDistance(),
			                                     fanprojgeom->getDetectorWidth(),
			                                     false); // TODO: Support short-scan

			iDetectorCount = fanprojgeom->getDetectorCount();
		} else {
			assert(false);
		}

		ok &= m_pFBP->setPixelSuperSampling(m_iPixelSuperSampling);

		ASTRA_ASSERT(ok);

		const float *pfTOffsets = m_pSinogram->getGeometry()->getExtraDetectorOffset();
		if (pfTOffsets)
			ok &= m_pFBP->setTOffsets(pfTOffsets);
		ASTRA_ASSERT(ok);

		ok &= m_pFBP->init(m_iGPUIndex);
		ASTRA_ASSERT(ok);

		ok &= m_pFBP->setSinogram(m_pSinogram->getDataConst(), iDetectorCount);
		ASTRA_ASSERT(ok);

		ok &= m_pFBP->setFilter(m_eFilter, m_pfFilter, m_iFilterWidth, m_fFilterD, m_fFilterParameter);
		ASTRA_ASSERT(ok);

		m_bAstraFBPInit = true;
	}

	bool ok = m_pFBP->run();
	ASTRA_ASSERT(ok);

	const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();
	ok &= m_pFBP->getReconstruction(m_pReconstruction->getData(), volgeom.getGridColCount());
	
	ASTRA_ASSERT(ok);
}

bool CCudaFilteredBackProjectionAlgorithm::check()
{
	// check pointers
	ASTRA_CONFIG_CHECK(m_pSinogram, "FBP_CUDA", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pReconstruction, "FBP_CUDA", "Invalid Reconstruction Data Object.");
	
	if((m_eFilter == FILTER_PROJECTION) || (m_eFilter == FILTER_SINOGRAM) || (m_eFilter == FILTER_RPROJECTION) || (m_eFilter == FILTER_RSINOGRAM))
	{
		ASTRA_CONFIG_CHECK(m_pfFilter, "FBP_CUDA", "Invalid filter pointer.");
	}

	// check initializations
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "FBP_CUDA", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isInitialized(), "FBP_CUDA", "Reconstruction Data Object Not Initialized.");

	// check gpu index
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= -1, "FBP_CUDA", "GPUIndex must be a non-negative integer.");
	// check pixel supersampling
	ASTRA_CONFIG_CHECK(m_iPixelSuperSampling >= 0, "FBP_CUDA", "PixelSuperSampling must be a non-negative integer.");


	// success
	m_bIsInitialized = true;
	return true;
}

static int calcNextPowerOfTwo(int _iValue)
{
	int iOutput = 1;

	while(iOutput < _iValue)
	{
		iOutput *= 2;
	}

	return iOutput;
}

int CCudaFilteredBackProjectionAlgorithm::calcIdealRealFilterWidth(int _iDetectorCount)
{
	return calcNextPowerOfTwo(_iDetectorCount);
}

int CCudaFilteredBackProjectionAlgorithm::calcIdealFourierFilterWidth(int _iDetectorCount)
{
	return (calcNextPowerOfTwo(_iDetectorCount) / 2 + 1);
}

static bool stringCompareLowerCase(const char * _stringA, const char * _stringB)
{
	int iCmpReturn = 0;

#ifdef _MSC_VER
	iCmpReturn = _stricmp(_stringA, _stringB);
#else
	iCmpReturn = strcasecmp(_stringA, _stringB);
#endif
	
	return (iCmpReturn == 0);
}

E_FBPFILTER CCudaFilteredBackProjectionAlgorithm::_convertStringToFilter(const char * _filterType)
{
	E_FBPFILTER output = FILTER_NONE;

	if(stringCompareLowerCase(_filterType, "ram-lak"))
	{
		output = FILTER_RAMLAK;
	}
	else if(stringCompareLowerCase(_filterType, "shepp-logan"))
	{
		output = FILTER_SHEPPLOGAN;
	}
	else if(stringCompareLowerCase(_filterType, "cosine"))
	{
		output = FILTER_COSINE;
	}
	else if(stringCompareLowerCase(_filterType, "hamming"))
	{
		output = FILTER_HAMMING;
	}
	else if(stringCompareLowerCase(_filterType, "hann"))
	{
		output = FILTER_HANN;
	}
	else if(stringCompareLowerCase(_filterType, "none"))
	{
		output = FILTER_NONE;
	}
	else if(stringCompareLowerCase(_filterType, "tukey"))
	{
		output = FILTER_TUKEY;
	}
	else if(stringCompareLowerCase(_filterType, "lanczos"))
	{
		output = FILTER_LANCZOS;
	}
	else if(stringCompareLowerCase(_filterType, "triangular"))
	{
		output = FILTER_TRIANGULAR;
	}
	else if(stringCompareLowerCase(_filterType, "gaussian"))
	{
		output = FILTER_GAUSSIAN;
	}
	else if(stringCompareLowerCase(_filterType, "barlett-hann"))
	{
		output = FILTER_BARTLETTHANN;
	}
	else if(stringCompareLowerCase(_filterType, "blackman"))
	{
		output = FILTER_BLACKMAN;
	}
	else if(stringCompareLowerCase(_filterType, "nuttall"))
	{
		output = FILTER_NUTTALL;
	}
	else if(stringCompareLowerCase(_filterType, "blackman-harris"))
	{
		output = FILTER_BLACKMANHARRIS;
	}
	else if(stringCompareLowerCase(_filterType, "blackman-nuttall"))
	{
		output = FILTER_BLACKMANNUTTALL;
	}
	else if(stringCompareLowerCase(_filterType, "flat-top"))
	{
		output = FILTER_FLATTOP;
	}
	else if(stringCompareLowerCase(_filterType, "kaiser"))
	{
		output = FILTER_KAISER;
	}
	else if(stringCompareLowerCase(_filterType, "parzen"))
	{
		output = FILTER_PARZEN;
	}
	else if(stringCompareLowerCase(_filterType, "projection"))
	{
		output = FILTER_PROJECTION;
	}
	else if(stringCompareLowerCase(_filterType, "sinogram"))
	{
		output = FILTER_SINOGRAM;
	}
	else if(stringCompareLowerCase(_filterType, "rprojection"))
	{
		output = FILTER_RPROJECTION;
	}
	else if(stringCompareLowerCase(_filterType, "rsinogram"))
	{
		output = FILTER_RSINOGRAM;
	}
	else
	{
		cerr << "Failed to convert \"" << _filterType << "\" into a filter." << endl;
	}

	return output;
}

void CCudaFilteredBackProjectionAlgorithm::testGenFilter(E_FBPFILTER _eFilter, float _fD, int _iProjectionCount, cufftComplex * _pFilter, int _iFFTRealDetectorCount, int _iFFTFourierDetectorCount)
{
	genFilter(_eFilter, _fD, _iProjectionCount, _pFilter, _iFFTRealDetectorCount, _iFFTFourierDetectorCount);
}

int CCudaFilteredBackProjectionAlgorithm::getGPUCount()
{
	return 0;
}
