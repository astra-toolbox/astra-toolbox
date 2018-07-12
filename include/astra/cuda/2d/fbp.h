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

#include "algo.h"
#include "astra/Filters.h"

namespace astraCUDA {

class _AstraExport FBP : public ReconAlgo {
public:
	FBP();
	~FBP();

	virtual bool useSinogramMask() { return false; }
	virtual bool useVolumeMask() { return false; }

	// Returns the required size of a filter in the fourier domain
	// when multiplying it with the fft of the projection data.
	// Its value is equal to the smallest power of two larger than
	// or equal to twice the number of detectors in the spatial domain.
	//
	// _iDetectorCount is the number of detectors in the spatial domain.
	static int calcFourierFilterSize(int _iDetectorCount);

	// Sets the filter type. Some filter types require the user to supply an
	// array containing the filter.
	// The number of elements in a filter in the fourier domain should be equal
	// to the value returned by calcFourierFilterSize().
	// The following types require a filter:
	//
	// - FILTER_PROJECTION:
	// The filter size should be equal to the output of
	// calcFourierFilterSize(). The filtered sinogram is
	// multiplied with the supplied filter.
	//
	// - FILTER_SINOGRAM:
	// Same as FILTER_PROJECTION, but now the filter should contain a row for
	// every projection direction.
	//
	// - FILTER_RPROJECTION:
	// The filter should now contain one kernel (= ifft of filter), with the
	// peak in the center. The filter width
	// can be any value. If odd, the peak is assumed to be in the center, if
	// even, it is assumed to be at floor(filter-width/2).
	//
	// - FILTER_RSINOGRAM
	// Same as FILTER_RPROJECTION, but now the supplied filter should contain a
	// row for every projection direction.
	//
	// A large number of other filters (FILTER_RAMLAK, FILTER_SHEPPLOGAN,
	// FILTER_COSINE, FILTER_HAMMING, and FILTER_HANN)
	// have a D variable, which gives the cutoff point in the frequency domain.
	// Setting this value to 1.0 will include the whole filter
	bool setFilter(const astra::SFilterConfig &_cfg);

	bool setShortScan(bool ss) { m_bShortScan = ss; return true; }

	virtual bool init();

	virtual bool iterate(unsigned int iterations);

	virtual float computeDiffNorm() { return 0.0f; } // TODO

protected:
	void reset();

	void* D_filter; // cufftComplex*
	bool m_bShortScan;
};

}
