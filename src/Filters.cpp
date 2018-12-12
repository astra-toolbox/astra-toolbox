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

#include "astra/Globals.h"
#include "astra/Logging.h"
#include "astra/Fourier.h"
#include "astra/Filters.h"
#include "astra/Config.h"
#include "astra/Float32ProjectionData2D.h"
#include "astra/AstraObjectManager.h"

#include <utility>
#include <cstring>

namespace astra {

float *genFilter(const SFilterConfig &_cfg,
               int _iFFTRealDetectorCount,
               int _iFFTFourierDetectorCount)
{
	float * pfFilt = new float[_iFFTFourierDetectorCount];
	float * pfW = new float[_iFFTFourierDetectorCount];

	// We cache one Fourier transform for repeated FBP's of the same size
	static float *pfData = 0;
	static int iFilterCacheSize = 0;

	if (!pfData || iFilterCacheSize != _iFFTRealDetectorCount) {
		// Compute filter in spatial domain

		delete[] pfData;
		pfData = new float[2*_iFFTRealDetectorCount];
		int *ip = new int[int(2+sqrt(_iFFTRealDetectorCount)+1)];
		ip[0] = 0;
		float32 *w = new float32[_iFFTRealDetectorCount/2];

		for (int i = 0; i < _iFFTRealDetectorCount; ++i) {
			pfData[2*i+1] = 0.0f;

			if (i & 1) {
				int j = i;
				if (2*j > _iFFTRealDetectorCount)
					j = _iFFTRealDetectorCount - j;
				float f = PI * j;
				pfData[2*i] = -1 / (f*f);
			} else {
				pfData[2*i] = 0.0f;
			}
		}

		pfData[0] = 0.25f;

		cdft(2*_iFFTRealDetectorCount, -1, pfData, ip, w);
		delete[] ip;
		delete[] w;

		iFilterCacheSize = _iFFTRealDetectorCount;
	}

	for(int iDetectorIndex = 0; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
	{
		float fRelIndex = (float)iDetectorIndex / (float)_iFFTRealDetectorCount;

		pfFilt[iDetectorIndex] = 2.0f * pfData[2*iDetectorIndex];
		pfW[iDetectorIndex] = PI * 2.0f * fRelIndex;
	}

	switch(_cfg.m_eType)
	{
		case FILTER_RAMLAK:
		{
			// do nothing
			break;
		}
		case FILTER_SHEPPLOGAN:
		{
			// filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)))
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * (sinf(pfW[iDetectorIndex] / 2.0f / _cfg.m_fD) / (pfW[iDetectorIndex] / 2.0f / _cfg.m_fD));
			}
			break;
		}
		case FILTER_COSINE:
		{
			// filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d))
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * cosf(pfW[iDetectorIndex] / 2.0f / _cfg.m_fD);
			}
			break;
		}
		case FILTER_HAMMING:
		{
			// filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d))
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * ( 0.54f + 0.46f * cosf(pfW[iDetectorIndex] / _cfg.m_fD));
			}
			break;
		}
		case FILTER_HANN:
		{
			// filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * (1.0f + cosf(pfW[iDetectorIndex] / _cfg.m_fD)) / 2.0f;
			}
			break;
		}
		case FILTER_TUKEY:
		{
			float fAlpha = _cfg.m_fParameter;
			if(_cfg.m_fParameter < 0.0f) fAlpha = 0.5f;
			float fN = (float)_iFFTFourierDetectorCount;
			float fHalfN = fN / 2.0f;
			float fEnumTerm = fAlpha * fHalfN;
			float fDenom = (1.0f - fAlpha) * fHalfN;
			float fBlockStart = fHalfN - fEnumTerm;
			float fBlockEnd = fHalfN + fEnumTerm;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fAbsSmallN = fabs((float)iDetectorIndex);
				float fStoredValue = 0.0f;

				if((fBlockStart <= fAbsSmallN) && (fAbsSmallN <= fBlockEnd))
				{
					fStoredValue = 1.0f;
				}
				else
				{
					float fEnum = fAbsSmallN - fEnumTerm;
					float fCosInput = PI * fEnum / fDenom;
					fStoredValue = 0.5f * (1.0f + cosf(fCosInput));
				}

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_LANCZOS:
		{
			float fDenum = (float)(_iFFTFourierDetectorCount - 1);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fX = 2.0f * fSmallN / fDenum - 1.0f;
				float fSinInput = PI * fX;
				float fStoredValue = 0.0f;

				if(fabsf(fSinInput) > 0.001f)
				{
					fStoredValue = sin(fSinInput)/fSinInput;
				}
				else
				{
					fStoredValue = 1.0f;
				}

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_TRIANGULAR:
		{
			float fNMinusOne = (float)(_iFFTFourierDetectorCount - 1);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fAbsInput = fSmallN - fNMinusOne / 2.0f;
				float fParenInput = fNMinusOne / 2.0f - fabsf(fAbsInput);
				float fStoredValue = 2.0f / fNMinusOne * fParenInput;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_GAUSSIAN:
		{
			float fSigma = _cfg.m_fParameter;
			if(_cfg.m_fParameter < 0.0f) fSigma = 0.4f;
			float fN = (float)_iFFTFourierDetectorCount;
			float fQuotient = (fN - 1.0f) / 2.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fEnum = fSmallN - fQuotient;
				float fDenom = fSigma * fQuotient;
				float fPower = -0.5f * (fEnum / fDenom) * (fEnum / fDenom);
				float fStoredValue = expf(fPower);

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BARTLETTHANN:
		{
			const float fA0 = 0.62f;
			const float fA1 = 0.48f;
			const float fA2 = 0.38f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fAbsInput = fSmallN / fNMinusOne - 0.5f;
				float fFirstTerm = fA1 * fabsf(fAbsInput);
				float fCosInput = 2.0f * PI * fSmallN / fNMinusOne;
				float fSecondTerm = fA2 * cosf(fCosInput);
				float fStoredValue = fA0 - fFirstTerm - fSecondTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BLACKMAN:
		{
			float fAlpha = _cfg.m_fParameter;
			if(_cfg.m_fParameter < 0.0f) fAlpha = 0.16f;
			float fA0 = (1.0f - fAlpha) / 2.0f;
			float fA1 = 0.5f;
			float fA2 = fAlpha / 2.0f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount - 1);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fCosInput1 = 2.0f * PI * 0.5f * fSmallN / fNMinusOne;
				float fCosInput2 = 4.0f * PI * 0.5f * fSmallN / fNMinusOne;
				float fStoredValue = fA0 - fA1 * cosf(fCosInput1) + fA2 * cosf(fCosInput2);

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_NUTTALL:
		{
			const float fA0 = 0.355768f;
			const float fA1 = 0.487396f;
			const float fA2 = 0.144232f;
			const float fA3 = 0.012604f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BLACKMANHARRIS:
		{
			const float fA0 = 0.35875f;
			const float fA1 = 0.48829f;
			const float fA2 = 0.14128f;
			const float fA3 = 0.01168f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BLACKMANNUTTALL:
		{
			const float fA0 = 0.3635819f;
			const float fA1 = 0.4891775f;
			const float fA2 = 0.1365995f;
			const float fA3 = 0.0106411f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_FLATTOP:
		{
			const float fA0 = 1.0f;
			const float fA1 = 1.93f;
			const float fA2 = 1.29f;
			const float fA3 = 0.388f;
			const float fA4 = 0.032f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fFourthTerm = fA4 * cosf(8.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm + fFourthTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_KAISER:
		{
			float fAlpha = _cfg.m_fParameter;
			if(_cfg.m_fParameter < 0.0f) fAlpha = 3.0f;
			float fPiTimesAlpha = PI * fAlpha;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount - 1);
			float fDenom = (float)j0((double)fPiTimesAlpha);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fSquareInput = 2.0f * fSmallN / fNMinusOne - 1;
				float fSqrtInput = 1.0f - fSquareInput * fSquareInput;
				float fBesselInput = fPiTimesAlpha * sqrt(fSqrtInput);
				float fEnum = (float)j0((double)fBesselInput);
				float fStoredValue = fEnum / fDenom;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_PARZEN:
		{
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fQ = fSmallN / (float)(_iFFTFourierDetectorCount - 1);
				float fStoredValue = 0.0f;

				if(fQ <= 0.5f)
				{
					fStoredValue = 1.0f - 6.0f * fQ * fQ * (1.0f - fQ);
				}
				else
				{
					float fCubedValue = 1.0f - fQ;
					fStoredValue = 2.0f * fCubedValue * fCubedValue * fCubedValue;
				}

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		default:
		{
			ASTRA_ERROR("Cannot serve requested filter");
		}
	}

	// filt(w>pi*d) = 0;
	float fPiTimesD = PI * _cfg.m_fD;
	for(int iDetectorIndex = 0; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
	{
		float fWValue = pfW[iDetectorIndex];

		if(fWValue > fPiTimesD)
		{
			pfFilt[iDetectorIndex] = 0.0f;
		}
	}

	delete[] pfW;

	return pfFilt;
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

struct FilterNameMapEntry {
	const char *m_name;
	E_FBPFILTER m_type;
};

E_FBPFILTER convertStringToFilter(const std::string &_filterType)
{

	static const FilterNameMapEntry map[] = {
		{ "ram-lak",           FILTER_RAMLAK },
		{ "shepp-logan",       FILTER_SHEPPLOGAN },
		{ "cosine",            FILTER_COSINE },
		{ "hamming",           FILTER_HAMMING},
		{ "hann",              FILTER_HANN},
		{ "tukey",             FILTER_TUKEY },
		{ "lanczos",           FILTER_LANCZOS},
		{ "triangular",        FILTER_TRIANGULAR},
		{ "gaussian",          FILTER_GAUSSIAN},
		{ "barlett-hann",      FILTER_BARTLETTHANN },
		{ "blackman",          FILTER_BLACKMAN},
		{ "nuttall",           FILTER_NUTTALL },
		{ "blackman-harris",   FILTER_BLACKMANHARRIS },
		{ "blackman-nuttall",  FILTER_BLACKMANNUTTALL },
		{ "flat-top",          FILTER_FLATTOP },
		{ "kaiser",            FILTER_KAISER },
		{ "parzen",            FILTER_PARZEN },
		{ "projection",        FILTER_PROJECTION },
		{ "sinogram",          FILTER_SINOGRAM },
		{ "rprojection",       FILTER_RPROJECTION },
		{ "rsinogram",         FILTER_RSINOGRAM },
		{ "none",              FILTER_NONE },
		{ 0,                   FILTER_ERROR } };

	const FilterNameMapEntry *i;

	for (i = &map[0]; i->m_name; ++i)
		if (stringCompareLowerCase(_filterType.c_str(), i->m_name))
			return i->m_type;

	ASTRA_ERROR("Failed to convert \"%s\" into a filter.",_filterType.c_str());

	return FILTER_ERROR;
}


SFilterConfig getFilterConfigForAlgorithm(const Config& _cfg, CAlgorithm *_alg)
{
	ConfigStackCheck<CAlgorithm> CC("getFilterConfig", _alg, _cfg);

	SFilterConfig c;

	XMLNode node;

	// filter type
	const char *nodeName = "FilterType";
	node = _cfg.self.getSingleNode(nodeName);
	if (_cfg.self.hasOption(nodeName)) {
		c.m_eType = convertStringToFilter(_cfg.self.getOption(nodeName));
		CC.markOptionParsed(nodeName);
	} else if (node) {
		// Fallback: check cfg.FilterType (instead of cfg.option.FilterType)
		c.m_eType = convertStringToFilter(node.getContent());
		CC.markNodeParsed(nodeName);
	} else {
		c.m_eType = FILTER_RAMLAK;
	}

	// filter
	nodeName = "FilterSinogramId";
	int id = -1;
	switch (c.m_eType) {
	case FILTER_PROJECTION:
	case FILTER_SINOGRAM:
	case FILTER_RPROJECTION:
	case FILTER_RSINOGRAM:
		node = _cfg.self.getSingleNode(nodeName);
		try {
			if (_cfg.self.hasOption(nodeName)) {
				id = _cfg.self.getOptionInt(nodeName);
				CC.markOptionParsed(nodeName);
			} else if (node) {
				id = node.getContentInt();
				CC.markNodeParsed(nodeName);
			}
		} catch (const astra::StringUtil::bad_cast &e) {
			ASTRA_ERROR("%s is not a valid id", nodeName);
		}
		break;
	default:
		break;
	}

	if (id != -1) {
		const CFloat32ProjectionData2D * pFilterData = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
		c.m_iCustomFilterWidth = pFilterData->getGeometry()->getDetectorCount();
		c.m_iCustomFilterHeight = pFilterData->getGeometry()->getProjectionAngleCount();

		c.m_pfCustomFilter = new float[c.m_iCustomFilterWidth * c.m_iCustomFilterHeight];
		memcpy(c.m_pfCustomFilter, pFilterData->getDataConst(), sizeof(float) * c.m_iCustomFilterWidth * c.m_iCustomFilterHeight);
	} else {
		c.m_iCustomFilterWidth = 0;
		c.m_iCustomFilterHeight = 0;
		c.m_pfCustomFilter = NULL;
	}

	// filter parameter
	nodeName = "FilterParameter";
	c.m_fParameter = -1.0f;
	switch (c.m_eType) {
	case FILTER_TUKEY:
	case FILTER_GAUSSIAN:
	case FILTER_BLACKMAN:
	case FILTER_KAISER:
		try {
			node = _cfg.self.getSingleNode(nodeName);
			if (_cfg.self.hasOption(nodeName)) {
				c.m_fParameter = _cfg.self.getOptionNumerical(nodeName);
				CC.markOptionParsed(nodeName);
			} else if (node) {
				c.m_fParameter = node.getContentNumerical();
				CC.markNodeParsed(nodeName);
			}
		} catch (const astra::StringUtil::bad_cast &e) {
			ASTRA_ERROR("%s must be numerical", nodeName);
		}
		break;
	default:
		break;
	}

	// D value
	nodeName = "FilterD";
	c.m_fD = 1.0f;
	switch (c.m_eType) {
	case FILTER_PROJECTION:
	case FILTER_SINOGRAM:
	case FILTER_RPROJECTION:
	case FILTER_RSINOGRAM:
		break;
	case FILTER_NONE:
	case FILTER_ERROR:
		break;
	default:
		try {
			node = _cfg.self.getSingleNode(nodeName);
			if (_cfg.self.hasOption(nodeName)) {
				c.m_fD = _cfg.self.getOptionNumerical(nodeName);
				CC.markOptionParsed(nodeName);
			} else if (node) {
				c.m_fD = node.getContentNumerical();
				CC.markNodeParsed(nodeName);
			}
		} catch (const astra::StringUtil::bad_cast &e) {
			ASTRA_ERROR("%s must be numerical", nodeName);
		}
		break;
	}
	return c;
}

int calcNextPowerOfTwo(int n)
{
	int x = 1;
	while (x < n && x > 0)
		x *= 2;

	return x;
}

// Because the input is real, the Fourier transform is symmetric.
// CUFFT only outputs the first half (ignoring the redundant second half),
// and expects the same as input for the IFFT.
int calcFFTFourierSize(int _iFFTRealSize)
{
	int iFFTFourierSize = _iFFTRealSize / 2 + 1;

	return iFFTFourierSize;
}

bool checkCustomFilterSize(const SFilterConfig &_cfg, const CProjectionGeometry2D &_geom) {
	int iExpectedWidth = -1, iExpectedHeight = 1;

	switch (_cfg.m_eType) {
		case FILTER_ERROR:
			ASTRA_ERROR("checkCustomFilterSize: internal error; FILTER_ERROR passed");
			return false;
		case FILTER_NONE:
			return true;
		case FILTER_SINOGRAM:
			iExpectedHeight = _geom.getProjectionAngleCount();
			// fallthrough
		case FILTER_PROJECTION:
			{
				int iPaddedDetCount = calcNextPowerOfTwo(2 * _geom.getDetectorCount());
				iExpectedWidth = calcFFTFourierSize(iPaddedDetCount);
			}
			if (_cfg.m_iCustomFilterWidth != iExpectedWidth ||
			    _cfg.m_iCustomFilterHeight != iExpectedHeight)
			{
				ASTRA_ERROR("filter size mismatch: %dx%d (received) is not %dx%d (expected)", _cfg.m_iCustomFilterHeight, _cfg.m_iCustomFilterWidth, iExpectedHeight, iExpectedWidth);
				return false;
			}
			return true;
		case FILTER_RSINOGRAM:
			iExpectedHeight = _geom.getProjectionAngleCount();
			// fallthrough
		case FILTER_RPROJECTION:
			if (_cfg.m_iCustomFilterHeight != iExpectedHeight)
			{
				ASTRA_ERROR("filter size mismatch: %dx%d (received) is not %dxX (expected)", _cfg.m_iCustomFilterHeight, _cfg.m_iCustomFilterWidth, iExpectedHeight);
				return false;
			}
			return true;
		default:
			// Non-custom filters; nothing to check.
			return true;
	}
}

}
