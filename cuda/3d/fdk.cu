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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/dims3d.h"
#include "astra/cuda/3d/arith3d.h"
#include "astra/cuda/3d/cone_bp.h"
#include "astra/cuda/3d/cone_bp_internal.h"

#include "astra/cuda/2d/fft.h"

#include "astra/cuda/stream.h"
#include "astra/cuda/stream_internal.h"

#include "astra/Logging.h"

#include <cstdio>
#include <cassert>
#include <vector>

namespace astraCUDA3d {

static const unsigned int g_anglesPerPrepareBlock = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

static const unsigned g_MaxAngles = 12000;

__constant__ float gC_angle[g_MaxAngles];

bool checkCufft(cufftResult err, const char *msg)
{
	if (err != CUFFT_SUCCESS) {
		ASTRA_ERROR("%s: CUFFT error %d.", msg, err);
		return false;
	} else {
		return true;
	}
}

struct ParkerWeightingData {
	double fCentralFanAngle = 0.0;
	double fCentralRayLength = 0.0;
	double fDetUSize = 0.0;
	double fScale = 0.0;
	double fAngleBase = 0.0;
};
struct PreWeightData {
	double fSrcOrigin = 0.0;
	double fDetOrigin = 0.0;
	double fZShift = 0.0;
	double fDetUSize = 0.0;
	double fDetVSize = 0.0;
};

class FDK_t : public ConeBP_t {
public:
	FDK_t()
	: m_D_filter(nullptr),
	  m_bSingleFilter(true),
	  m_D_pcSinoFFT(nullptr),
	  m_D_pfPadded(nullptr),
	  m_bShortScan(false),
	  m_D_pfParkerAngles(nullptr)
	{ }

	bool prepareFilter(const SDimensions3D& dims,
	                   const astra::SFilterConfig &filterConfig,
	                   cudaStream_t stream);

	void freeBuffers();

	bool prepareWeighting(bool bShortScan,
	                      const std::vector<SConeProjection>& angles,
	                      const SDimensions3D& dims,
	                      const SProjectorParams3D& params);

	virtual bool prepareProjections(cudaPitchedPtr D_tmpProjBuffer,
	                                cudaPitchedPtr D_projData,
	                                const SConeProjection *D_geom,
	                                unsigned int startAngle,
	                                const SDimensions3D& bufDims,
	                                const SDimensions3D& dims,
	                                const SProjectorParams3D& params,
	                                cudaStream_t stream) override;

private:
	bool filter(cudaPitchedPtr D_projData,
	            unsigned int startAngle,
	            const SDimensions3D& bufdims, const SDimensions3D& dims,
	            const SProjectorParams3D& params,
	            cudaStream_t stream);

	bool doParkerWeighting(cudaPitchedPtr D_projData,
	                       unsigned int startAngle,
	                       const SDimensions3D& bufdims, const SDimensions3D& dims,
	                       cudaStream_t stream);

	cufftComplex *m_D_filter;
	bool m_bSingleFilter;

	cufftComplex * m_D_pcSinoFFT;
	float * m_D_pfPadded;

	cufftHandle m_planF;
	cufftHandle m_planI;

	// Parker weighting parameters
	bool m_bShortScan;
	ParkerWeightingData m_Parker;
	float * m_D_pfParkerAngles;

	// Pre-weighting parameters
	PreWeightData m_preweight;
};


__global__ void devFDK_preweight(float* D_outData, const float* D_projData, unsigned int projPitchOut, unsigned int projPitchIn, unsigned int startAngle, float fSrcOrigin, float fDetOrigin, float fZShift, float fDetUSize, float fDetVSize, const SDimensions3D bufdims, const SDimensions3D dims)
{
	int angle = blockIdx.z * blockDim.y + threadIdx.y;
	if (startAngle + angle >= dims.iProjAngles)
		return;
	if (angle >= bufdims.iProjAngles)
		return;

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	const int startDetectorV = blockIdx.y * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;


	// We need the length of the central ray and the length of the ray(s) to
	// our detector pixel(s).

	const float fCentralRayLength = fSrcOrigin + fDetOrigin;

	const float fU = (detectorU - 0.5f*dims.iProjU + 0.5f) * fDetUSize;

	const float fT = fCentralRayLength * fCentralRayLength + fU * fU;

	float fV = (startDetectorV - 0.5f*dims.iProjV + 0.5f) * fDetVSize + fZShift;

	// Contributions to the weighting factors:
	// fCentralRayLength / fRayLength   : the main FDK preweighting factor
	// fCentralRayLength / (fDetUSize * fSrcOrigin)
	//                                  : to adjust the filter to the det width
	// pi / (2 * iProjAngles)           : scaling of the integral over angles

	const float fW2 = fCentralRayLength / (fDetUSize * fSrcOrigin);
	const float fW = fCentralRayLength * fW2 * (M_PI / 2.0f) / (float)dims.iProjAngles;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fRayLength = sqrtf(fT + fV * fV);

		const float fWeight = fW / fRayLength;

		D_outData[(size_t)(angle*bufdims.iProjV+detectorV)*projPitchOut+detectorU] = D_projData[(size_t)(detectorV*dims.iProjAngles+startAngle+angle)*projPitchIn+detectorU] * fWeight;

		fV += fDetVSize;
	}
}

__global__ void devFDK_ParkerWeight(float* D_projData, unsigned int projPitch, const float* D_relAngles, unsigned int startAngle,
		float fCentralRayLength, float fDetUSize, float fCentralFanAngle, float fScale, const SDimensions3D bufDims, unsigned int totalAngles)
{
	int angle = blockIdx.z * blockDim.y + threadIdx.y;
	if (startAngle + angle >= totalAngles)
		return;
	if (angle >= bufDims.iProjAngles)
		return;

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	const int startDetectorV = blockIdx.y * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > bufDims.iProjV)
		endDetectorV = bufDims.iProjV;

	// We need the length of the central ray and the length of the projection
	// of our ray onto the central slice to compute gamma

	// TODO: Detector pixel size
	const float fU = (detectorU - 0.5f*bufDims.iProjU + 0.5f) * fDetUSize;

	const float fGamma = atanf(fU / fCentralRayLength);
	float fBeta = D_relAngles[startAngle+angle];

	// compute the weight depending on the location in the central fan's radon
	// space
	float fWeight;

	if (fBeta <= 0.0f) {
		fWeight = 0.0f;
	} else if (fBeta <= 2.0f*(fCentralFanAngle + fGamma)) {
		fWeight = sinf((M_PI / 4.0f) * fBeta / (fCentralFanAngle + fGamma));
		fWeight *= fWeight;
	} else if (fBeta <= M_PI + 2*fGamma) {
		fWeight = 1.0f;
	} else if (fBeta <= M_PI + 2*fCentralFanAngle) {
		fWeight = sinf((M_PI / 4.0f) * (M_PI + 2.0f*fCentralFanAngle - fBeta) / (fCentralFanAngle - fGamma));
		fWeight *= fWeight;
	} else {
		fWeight = 0.0f;
	}

	fWeight *= fScale;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		D_projData[(size_t)(angle*bufDims.iProjV+detectorV)*projPitch+detectorU] *= fWeight;
	}
}

bool FDK_t::doParkerWeighting(cudaPitchedPtr D_projData,
	                      unsigned int startAngle,
	                      const SDimensions3D& bufdims, const SDimensions3D& dims,
	                      cudaStream_t stream)
{
	if (!m_bShortScan)
		return true;

	dim3 dimBlock(g_detBlockU, g_anglesPerPrepareBlock);
	dim3 dimGrid = computeGrid3Ceil(dims.iProjU, g_detBlockU,
	                                dims.iProjV, g_detBlockV,
	                                bufdims.iProjAngles, g_anglesPerPrepareBlock);

	int projPitch = D_projData.pitch/sizeof(float);

	devFDK_ParkerWeight<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, projPitch, m_D_pfParkerAngles, startAngle, m_Parker.fCentralRayLength, m_Parker.fDetUSize, m_Parker.fCentralFanAngle, m_Parker.fScale, bufdims, dims.iProjAngles);

	return true;
}

bool FDK_t::filter(cudaPitchedPtr D_projData,
                   unsigned int startAngle,
                   const SDimensions3D& bufdims, const SDimensions3D& dims,
                   const SProjectorParams3D& params,
                   cudaStream_t stream)
{
	if (!m_D_filter)
		return true;

	// Filter the current subset of projections
	unsigned int angleCount = bufdims.iProjAngles;
	if (startAngle + angleCount > dims.iProjAngles)
		angleCount = dims.iProjAngles - startAngle;

	int iPaddedDetCount = calcNextPowerOfTwo(2 * dims.iProjU);
	size_t bufferMemSize = sizeof(float) * dims.iProjV * iPaddedDetCount;
	int iHalfFFTSize = astra::calcFFTFourierSize(iPaddedDetCount);

	// We process one projection image at a time
	float* D_imgData = (float*)D_projData.ptr;
	cufftComplex* D_filter = m_D_filter;
	if (!m_bSingleFilter)
		D_filter += startAngle * iHalfFFTSize;
	bool ok = true;
	for (unsigned int angle = 0; angle < angleCount; ++angle)
	{
		if (!checkCuda(cudaMemsetAsync(m_D_pfPadded, 0, bufferMemSize, stream), "FDK filter pre memset")) {
			ok = false;
			break;
		}

		// pitched memcpy 2D to handle both source pitch and target padding
		if (!checkCuda(cudaMemcpy2DAsync(m_D_pfPadded, iPaddedDetCount*sizeof(float), D_imgData, D_projData.pitch, dims.iProjU*sizeof(float), dims.iProjV, cudaMemcpyDeviceToDevice, stream), "FDK filter memcpy")) {
			ok = false;
			break;
		}

		if (!checkCufft(cufftExecR2C(m_planF, (cufftReal *)m_D_pfPadded, m_D_pcSinoFFT), "FDK filter forward exec")) {
			ok = false;
			break;
		}

		// Always call the 2D applyFilter with singleFilter = true,
		// since we are calling it on a single projection image, and
		// not on a sinogram.
		if (!astraCUDA::applyFilter(dims.iProjV, iHalfFFTSize, m_D_pcSinoFFT, D_filter, true, stream)) {
			ok = false;
			break;
		}

		// Getting rid of the const qualifier is due to cufft API issue?
		if (!checkCufft(cufftExecC2R(m_planI, (cufftComplex *)m_D_pcSinoFFT,
	                      (cufftReal *)m_D_pfPadded), "FDK filter inverse exec"))
		{
			ok = false;
			break;
		}

		if (!checkCuda(cudaMemsetAsync(D_imgData, 0, dims.iProjV * D_projData.pitch, stream), "FDK filter post memset")) {
			ok = false;
			break;
		}

		// pitched memcpy 2D to handle both source padding and target pitch
		if (!checkCuda(cudaMemcpy2DAsync(D_imgData, D_projData.pitch, m_D_pfPadded, iPaddedDetCount*sizeof(float), dims.iProjU*sizeof(float), dims.iProjV, cudaMemcpyDeviceToDevice, stream), "FDK filter memcpy")) {
			ok = false;
			break;
		}

		D_imgData += dims.iProjV * (D_projData.pitch / sizeof(float));
		if (!m_bSingleFilter)
			D_filter += iHalfFFTSize;
	}

	ok &= checkCuda(cudaStreamSynchronize(stream), "FDK filter sync");

	return ok;
}

void FDK_t::freeBuffers()
{
	if (m_D_filter) {
		cufftDestroy(m_planF);
		cufftDestroy(m_planI);

		logCuda(cudaFree(m_D_pfPadded), "FDK filter free");
		astraCUDA::freeComplexOnDevice(m_D_pcSinoFFT);
		astraCUDA::freeComplexOnDevice(m_D_filter);
	}

	if (m_bShortScan) {
		logCuda(cudaFree(m_D_pfParkerAngles), "FDK shortscan free");
	}
}

bool FDK_t::prepareProjections(cudaPitchedPtr D_outData, cudaPitchedPtr D_projData,
                               const SConeProjection* D_geom,
                               unsigned int startAngle,
                               const SDimensions3D& bufdims, const SDimensions3D& dims,
                               const SProjectorParams3D& params,
                               cudaStream_t stream)
{
	dim3 dimBlock(g_detBlockU, g_anglesPerPrepareBlock);
	dim3 dimGrid = computeGrid3Ceil(dims.iProjU, g_detBlockU,
	                                dims.iProjV, g_detBlockV,
	                                bufdims.iProjAngles, g_anglesPerPrepareBlock);

	int projPitchOut = D_outData.pitch/sizeof(float);
	int projPitchIn = D_projData.pitch/sizeof(float);

	devFDK_preweight<<<dimGrid, dimBlock, 0, stream>>>((float*)D_outData.ptr, (const float*)D_projData.ptr, projPitchOut, projPitchIn, startAngle, m_preweight.fSrcOrigin, m_preweight.fDetOrigin, m_preweight.fZShift, m_preweight.fDetUSize, m_preweight.fDetVSize, bufdims, dims);

	bool ok = doParkerWeighting(D_outData, startAngle, bufdims, dims, stream);
	if (!ok)
		return false;

	ok &= filter(D_outData, startAngle, bufdims, dims, params, stream);

	return ok;
}

bool FDK_t::prepareFilter(const SDimensions3D& dims,
                          const astra::SFilterConfig &filterConfig,
                          cudaStream_t stream)
{
	cufftComplex *D_filter = nullptr;
	bool singleFilter = true;

	bool ok = astraCUDA::prepareCuFFTFilter(filterConfig, D_filter, singleFilter, dims.iProjAngles, dims.iProjU, stream);

	if (!ok) {
		if (D_filter)
			astraCUDA::freeComplexOnDevice(D_filter);
		return false;
	}

	if (!D_filter) {
		// No filtering step requested
		m_D_filter = nullptr;
		return true;
	}

	// We pre-plan the FFTs for a projection image.

	int iPaddedDetCount = calcNextPowerOfTwo(2 * dims.iProjU);
	int iHalfFFTSize = astra::calcFFTFourierSize(iPaddedDetCount);

	if (!checkCufft(cufftPlan1d(&m_planF, iPaddedDetCount, CUFFT_R2C, dims.iProjV), "FDK filter FFT plan")) {
		if (D_filter)
			astraCUDA::freeComplexOnDevice(D_filter);
		return false;
	}
	if (!checkCufft(cufftSetStream(m_planF, stream), "FDK filter FFT plan stream")) {
		if (D_filter)
			astraCUDA::freeComplexOnDevice(D_filter);
		cufftDestroy(m_planF);
		return false;
	}

	if (!checkCufft(cufftPlan1d(&m_planI, iPaddedDetCount, CUFFT_C2R, dims.iProjV), "FDK filter IFFT plan")) {
		if (D_filter)
			astraCUDA::freeComplexOnDevice(D_filter);
		cufftDestroy(m_planF);
		return false;
	}
	if (!checkCufft(cufftSetStream(m_planI, stream), "FDK filter IFFT plan stream")) {
		if (D_filter)
			astraCUDA::freeComplexOnDevice(D_filter);
		cufftDestroy(m_planF);
		cufftDestroy(m_planI);
		return false;
	}

	// Pre-allocate FFT buffers
	cufftComplex * D_pcSinoFFT = nullptr;
	if (!astraCUDA::allocateComplexOnDevice(dims.iProjV, iHalfFFTSize, &D_pcSinoFFT)) {
		astraCUDA::freeComplexOnDevice(D_filter);
		cufftDestroy(m_planF);
		cufftDestroy(m_planI);
		return false;
	}

	float * D_pfPadded = NULL;
	size_t bufferMemSize = sizeof(float) * dims.iProjV * iPaddedDetCount;
	if (!checkCuda(cudaMalloc((void **)&D_pfPadded, bufferMemSize), "FDK filter malloc")) {
		astraCUDA::freeComplexOnDevice(D_pcSinoFFT);
		astraCUDA::freeComplexOnDevice(D_filter);
		cufftDestroy(m_planF);
		cufftDestroy(m_planI);
		return false;
	}

	m_D_filter = D_filter;
	m_bSingleFilter = singleFilter;
	m_D_pcSinoFFT = D_pcSinoFFT;
	m_D_pfPadded = D_pfPadded;

	return true;
}

bool FDK_t::prepareWeighting(bool bShortScan,
                             const std::vector<SConeProjection>& angles,
                             const SDimensions3D& dims,
                             const SProjectorParams3D& params)
{
	assert(angles.size() == dims.iProjAngles);
	if (dims.iProjAngles <= 1)
		bShortScan = false;

	m_bShortScan = bShortScan;

	const SConeProjection& p0 = angles[0];

	// NB: We don't support arbitrary cone_vec geometries here.
	// Only those that are vertical sub-geometries
	// (cf. CompositeGeometryManager) of regular cone geometries.

	// assuming U is in the XY plane, V is parallel to Z axis
	double fDetCX = p0.fDetSX + 0.5*dims.iProjU*p0.fDetUX;
	double fDetCY = p0.fDetSY + 0.5*dims.iProjU*p0.fDetUY;
	double fDetCZ = p0.fDetSZ + 0.5*dims.iProjV*p0.fDetVZ;

	m_preweight.fSrcOrigin = sqrt(p0.fSrcX*p0.fSrcX + p0.fSrcY*p0.fSrcY);
	m_preweight.fDetOrigin = sqrt(fDetCX*fDetCX + fDetCY*fDetCY);

	m_preweight.fDetUSize = sqrt(p0.fDetUX*p0.fDetUX + p0.fDetUY*p0.fDetUY);
	m_preweight.fDetVSize = std::abs(p0.fDetVZ);

	m_preweight.fZShift = fDetCZ - p0.fSrcZ;


	if (!m_bShortScan)
		return true;

	ASTRA_DEBUG("Doing Parker weighting");
	// We do short-scan Parker weighting

	m_Parker.fDetUSize = m_preweight.fDetUSize;

	m_Parker.fCentralRayLength = m_preweight.fSrcOrigin + m_preweight.fDetOrigin;



	std::vector<double> circAngles;
	circAngles.resize(dims.iProjAngles);

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		circAngles[i] = -atan2(angles[i].fSrcX, angles[i].fSrcY) + M_PI;


	// First, determine (in a very basic way) the interval that's
	// been scanned. We assume angles[0] is one of the endpoints of the
	// range.
	float fdA = circAngles[1] - circAngles[0];

	while (fdA < -M_PI)
		fdA += 2*M_PI;
	while (fdA >= M_PI)
		fdA -= 2*M_PI;

	if (fdA >= 0.0f) {
		// going up from angles[0]
		m_Parker.fAngleBase = circAngles[0];
		ASTRA_DEBUG("Second angle >= first angle, so assuming angles are incrementing");
	} else {
		// going down from angles[0]
		m_Parker.fAngleBase = circAngles[dims.iProjAngles - 1];
		ASTRA_DEBUG("Second angle < first angle, so assuming angles are decrementing");
	}

	// We pick the lowest end of the range, and then
	// move all angles so they fall in [0,2pi)

	std::vector<float> relAngles(dims.iProjAngles);
	for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
		double f = circAngles[i] - m_Parker.fAngleBase;
		while (f >= 2*M_PI)
			f -= 2*M_PI;
		while (f < 0)
			f += 2*M_PI;
		relAngles[i] = f;
	}

	double fRange = fabs(relAngles[dims.iProjAngles-1] - relAngles[0]);
	// Adjust for discretisation
	fRange /= dims.iProjAngles - 1;
	fRange *= dims.iProjAngles;

	ASTRA_DEBUG("Assuming angles are linearly ordered and equally spaced for Parker weighting. Angular range %f radians", fRange);
	m_Parker.fScale = fRange / M_PI;

	m_Parker.fCentralFanAngle = fabs(atan(m_Parker.fDetUSize * (dims.iProjU*0.5f) /
	                               m_Parker.fCentralRayLength));

	if (fRange + 1e-3 < M_PI + 2*m_Parker.fCentralFanAngle) {
		ASTRA_WARN("Angular range (%f rad) smaller than Parker weighting range (%f rad)", fRange, M_PI + 2*m_Parker.fCentralFanAngle);
	}

	// Upload relAngles array to GPU memory
	if (!checkCuda(cudaMalloc((void**)&m_D_pfParkerAngles, sizeof(float) * dims.iProjAngles), "prepareWeighting alloc")) {
		return false;
	}

	// TODO: async

	if (!checkCuda(cudaMemcpy(m_D_pfParkerAngles, &relAngles[0], sizeof(float) * dims.iProjAngles, cudaMemcpyHostToDevice), "prepareWeighting memcpy")) {
		logCuda(cudaFree(m_D_pfParkerAngles), "prepareWeighting free buffer");
		return false;
	}

	return true;
}


bool FDK(cudaPitchedPtr D_volumeData,
         cudaPitchedPtr D_projData,
         const std::vector<SConeProjection>& angles,
         const SDimensions3D& dims, SProjectorParams3D params_, bool bShortScan,
         const astra::SFilterConfig &filterConfig)
{
	Stream stream;
	SProjectorParams3D params = params_;

	FDK_t F;

	bool ok = true;
	ok = F.prepareWeighting(bShortScan, angles, dims, params);
	if (!ok)
		return false;

	ok = F.prepareFilter(dims, filterConfig, **stream);
	if (!ok) {
		F.freeBuffers();
		return false;
	}

	if (filterConfig.m_eType != astra::FILTER_NONE) {
		// HACK: when we do filtering, we introduce an extra factor of iPaddedDetCount
		// from the fft, ifft combination. We want to cancel that.
		int iPaddedDetCount = calcNextPowerOfTwo(2 * dims.iProjU);
		params.fOutputScale /= iPaddedDetCount;
	}

	ok = F.project(D_volumeData, D_projData, ConeBP_t::MODE_BACKWARD, false, dims, angles, params, stream);

	// TODO: superfluous with current sync at end of project() ?
	ok &= stream.sync("FDK sync");

	F.freeBuffers();

	return ok;
}

}
