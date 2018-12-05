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

#ifndef _INC_ASTRA_FILTERS_H
#define _INC_ASTRA_FILTERS_H

#include <string>

namespace astra {

struct Config;
class CAlgorithm;
class CProjectionGeometry2D;

enum E_FBPFILTER
{
	FILTER_ERROR,			//< not a valid filter
	FILTER_NONE,			//< no filter (regular BP)
	FILTER_RAMLAK,			//< default FBP filter
	FILTER_SHEPPLOGAN,		//< Shepp-Logan
	FILTER_COSINE,			//< Cosine
	FILTER_HAMMING,			//< Hamming filter
	FILTER_HANN,			//< Hann filter
	FILTER_TUKEY,			//< Tukey filter
	FILTER_LANCZOS,			//< Lanczos filter
	FILTER_TRIANGULAR,		//< Triangular filter
	FILTER_GAUSSIAN,		//< Gaussian filter
	FILTER_BARTLETTHANN,	//< Bartlett-Hann filter
	FILTER_BLACKMAN,		//< Blackman filter
	FILTER_NUTTALL,			//< Nuttall filter, continuous first derivative
	FILTER_BLACKMANHARRIS,	//< Blackman-Harris filter
	FILTER_BLACKMANNUTTALL,	//< Blackman-Nuttall filter
	FILTER_FLATTOP,			//< Flat top filter
	FILTER_KAISER,			//< Kaiser filter
	FILTER_PARZEN,			//< Parzen filter
	FILTER_PROJECTION,		//< all projection directions share one filter
	FILTER_SINOGRAM,		//< every projection direction has its own filter
	FILTER_RPROJECTION,		//< projection filter in real space (as opposed to fourier space)
	FILTER_RSINOGRAM,		//< sinogram filter in real space

};

struct SFilterConfig {
	E_FBPFILTER m_eType;
	float m_fD;
	float m_fParameter;

	float *m_pfCustomFilter;
	int m_iCustomFilterWidth;
	int m_iCustomFilterHeight;

	SFilterConfig() : m_eType(FILTER_ERROR), m_fD(1.0f), m_fParameter(-1.0f),
	                  m_pfCustomFilter(0), m_iCustomFilterWidth(0),
	                  m_iCustomFilterHeight(0) { };
};

// Generate filter of given size and parameters. Returns newly allocated array.
float *genFilter(const SFilterConfig &_cfg,
                 int _iFFTRealDetectorCount,
                 int _iFFTFourierDetectorCount);

// Convert string to filter type. Returns FILTER_ERROR if unrecognized.
E_FBPFILTER convertStringToFilter(const std::string &_filterType);

SFilterConfig getFilterConfigForAlgorithm(const Config& _cfg, CAlgorithm *_alg);

bool checkCustomFilterSize(const SFilterConfig &_cfg, const CProjectionGeometry2D &_geom);


int calcNextPowerOfTwo(int _iValue);
int calcFFTFourierSize(int _iFFTRealSize);


}

#endif
