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

#ifndef _CUDA_ASTRA_H
#define _CUDA_ASTRA_H

#include "fft.h"
#include "fbp_filters.h"
#include "dims.h"
#include "algo.h"

using astraCUDA::SFanProjection;

namespace astra {

enum Cuda2DProjectionKernel {
	ker2d_default = 0
};

class CParallelProjectionGeometry2D;
class CFanFlatProjectionGeometry2D;
class CFanFlatVecProjectionGeometry2D;
class CVolumeGeometry2D;

class AstraFBP_internal;

class _AstraExport AstraFBP {
public:
	// Constructor
	AstraFBP();

	// Destructor
	~AstraFBP();

	// Set the size of the reconstruction rectangle.
	// Volume pixels are currently assumed to be 1x1 squares.
	bool setReconstructionGeometry(unsigned int iVolWidth,
	                               unsigned int iVolHeight,
	                               float fPixelSize = 1.0f);

	// Set the projection angles and number of detector pixels per angle.
	// pfAngles must be a float array of length iProjAngles.
	// fDetSize indicates the size of a detector pixel compared to a
	// volume pixel edge.
	//
	// pfAngles will only be read from during this call.
	bool setProjectionGeometry(unsigned int iProjAngles,
	                           unsigned int iProjDets,
	                           const float *pfAngles,
	                           float fDetSize = 1.0f);
	// Set the projection angles and number of detector pixels per angle.
	// pfAngles must be a float array of length iProjAngles.
	// fDetSize indicates the size of a detector pixel compared to a
	// volume pixel edge.
	//
	// pfAngles, fanProjs will only be read from during this call.
	bool setFanGeometry(unsigned int iProjAngles,
	                    unsigned int iProjDets,
	                    const astraCUDA::SFanProjection *fanProjs,
	                    const float *pfAngles,
	                    float fOriginSourceDistance,
	                    float fOriginDetectorDistance,
	                    float fDetSize = 1.0f,
	                    bool bShortScan = false);

	// Set linear supersampling factor for the BP.
	// (The number of rays is the square of this)
	//
	// This may optionally be called before init().
	bool setPixelSuperSampling(unsigned int iPixelSuperSampling);

	// Set per-detector shifts.
	//
	// pfTOffsets will only be read from during this call.
	bool setTOffsets(const float *pfTOffsets);

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
	bool setFilter(E_FBPFILTER _eFilter,
                   const float * _pfHostFilter = NULL,
                   int _iFilterWidth = 0, float _fD = 1.0f, float _fFilterParameter = -1.0f);

	// Initialize CUDA, allocate GPU buffers and
	// precompute geometry-specific data.
	//
	// CUDA is set up to use GPU number iGPUIndex.
	//
	// This must be called after calling setReconstructionGeometry() and
	// setProjectionGeometry().
	bool init(int iGPUIndex = 0);

	// Setup input sinogram for a slice.
	// pfSinogram must be a float array of size iProjAngles*iSinogramPitch.
	// NB: iSinogramPitch is measured in floats, not in bytes.
	//
	// This must be called after init(), and before iterate(). It may be
	// called again after iterate()/getReconstruction() to start a new slice.
	//
	// pfSinogram will only be read from during this call.
	bool setSinogram(const float* pfSinogram, unsigned int iSinogramPitch);

	// Runs an FBP reconstruction.
	// This must be called after setSinogram().
	//
	// run can be called before setFilter, but will then use the default Ram-Lak filter
	bool run();

	// Get the reconstructed slice.
	// pfReconstruction must be a float array of size
	// iVolHeight*iReconstructionPitch.
	// NB: iReconstructionPitch is measured in floats, not in bytes.
	//
	// This may be called after run().
	bool getReconstruction(float* pfReconstruction,
	                       unsigned int iReconstructionPitch) const;

private:
	AstraFBP_internal* pData;
};

class _AstraExport BPalgo : public astraCUDA::ReconAlgo {
public:
	BPalgo();
	~BPalgo();

	virtual bool init();

	virtual bool iterate(unsigned int iterations);

	virtual float computeDiffNorm();
};




// TODO: Clean up this interface to FP

// Do a single forward projection
_AstraExport bool astraCudaFP(const float* pfVolume, float* pfSinogram,
                 unsigned int iVolWidth, unsigned int iVolHeight,
                 unsigned int iProjAngles, unsigned int iProjDets,
                 const float *pfAngles, const float *pfOffsets,
                 float fDetSize = 1.0f, unsigned int iDetSuperSampling = 1,
                 float fOutputScale = 1.0f, int iGPUIndex = 0);

_AstraExport bool astraCudaFanFP(const float* pfVolume, float* pfSinogram,
                    unsigned int iVolWidth, unsigned int iVolHeight,
                    unsigned int iProjAngles, unsigned int iProjDets,
                    const SFanProjection *pAngles,
                    unsigned int iDetSuperSampling = 1,
                    float fOutputScale = 1.0f, int iGPUIndex = 0);


_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CParallelProjectionGeometry2D* pProjGeom,
                    float*& pfDetectorOffsets, float*& pfProjectionAngles,
                    float& fDetSize, float& fOutputScale);

_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CFanFlatProjectionGeometry2D* pProjGeom,
                    astraCUDA::SFanProjection*& pProjs,
                    float& outputScale);

_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CFanFlatVecProjectionGeometry2D* pProjGeom,
                    astraCUDA::SFanProjection*& pProjs,
                    float& outputScale);


}
#endif
