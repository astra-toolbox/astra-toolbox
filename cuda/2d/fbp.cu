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

#include "astra/cuda/2d/fbp.h"
#include "astra/cuda/2d/fft.h"
#include "astra/cuda/2d/par_bp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/mem2d.h"
#include "astra/cuda/2d/util.h"
#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Logging.h"
#include "astra/Filters.h"
#include "astra/Data2D.h"

namespace astraCUDA {

// parameters for preweighting
static const unsigned int g_anglesPerWeightBlock = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;
static const unsigned g_MaxAngles = 12000;
__constant__ float gC_angle[g_MaxAngles];
// ----


struct SFilter_internal {
	cufftComplex *D_filter;
	bool single;
};

SFilter_internal *prepareFilter(const astra::SFilterConfig &_cfg, const SDimensions &dims)
{
	bool singleFilter;
	cufftComplex *f;

	bool ok = prepareCuFFTFilter(_cfg, f, singleFilter, dims.iProjAngles, dims.iProjDets);

	if (!ok)
		return nullptr;

	SFilter_internal *sf = new SFilter_internal;
	sf->D_filter = f;
	sf->single = singleFilter;

	return sf;
}

void freeFilter(SFilter_internal *f)
{
	assert(f);
	if (f->D_filter)
		freeComplexOnDevice(f->D_filter);
	delete f;
}


// ------------- start of code borrowed from FDK -------------------
// TODO: Simplify this into pure 2D code?
// (The 3D FDK code has diverged from this since it has moved to a different memory layout)

__global__ void devFBP_preweight(void* D_projData, unsigned int projPitch, unsigned int startAngle, unsigned int endAngle, float fSrcOrigin, float fDetOrigin, float fZShift, float fDetUSize, float fDetVSize, const astraCUDA3d::SDimensions3D dims)
{
	float* projData = (float*)D_projData;
	int angle = startAngle + blockIdx.y * g_anglesPerWeightBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
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

		projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] *= fWeight;

		fV += fDetVSize;
	}
}

__global__ void devFBP_ParkerWeight(void* D_projData, unsigned int projPitch, unsigned int startAngle, unsigned int endAngle, float fSrcOrigin, float fDetOrigin, float fDetUSize, float fCentralFanAngle, float fScale, const astraCUDA3d::SDimensions3D dims)
{
	float* projData = (float*)D_projData;
	int angle = startAngle + blockIdx.y * g_anglesPerWeightBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	// We need the length of the central ray and the length of the projection
	// of our ray onto the central slice

	const float fCentralRayLength = fSrcOrigin + fDetOrigin;

	// TODO: Detector pixel size
	const float fU = (detectorU - 0.5f*dims.iProjU + 0.5f) * fDetUSize;

	const float fGamma = atanf(fU / fCentralRayLength);
	float fBeta = gC_angle[angle];

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
		projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] *= fWeight;
	}
}



// Perform the FDK pre-weighting and filtering
bool FBP_PreWeight(cudaPitchedPtr D_projData,
                float fSrcOrigin, float fDetOrigin,
                float fZShift,
                float fDetUSize, float fDetVSize,
                bool bShortScan,
                const astraCUDA3d::SDimensions3D& dims, const float* angles)
{
	// The pre-weighting factor for a ray is the cosine of the angle between
	// the central line and the ray.

	cudaStream_t stream;
	if (!checkCuda(cudaStreamCreate(&stream), "FBP_PreWeight stream"))
		return false;

	dim3 dimBlock(g_detBlockU, g_anglesPerWeightBlock);
	dim3 dimGrid( ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
	              (dims.iProjAngles+g_anglesPerWeightBlock-1)/g_anglesPerWeightBlock);

	int projPitch = D_projData.pitch/sizeof(float);

	devFBP_preweight<<<dimGrid, dimBlock, 0, stream>>>(D_projData.ptr, projPitch, 0, dims.iProjAngles, fSrcOrigin, fDetOrigin, fZShift, fDetUSize, fDetVSize, dims);

	if (bShortScan && dims.iProjAngles > 1) {
		ASTRA_DEBUG("Doing Parker weighting");
		// We do short-scan Parker weighting

		// First, determine (in a very basic way) the interval that's
		// been scanned. We assume angles[0] is one of the endpoints of the
		// range.
		float fdA = angles[1] - angles[0];

		while (fdA < -M_PI)
			fdA += 2*M_PI;
		while (fdA >= M_PI)
			fdA -= 2*M_PI;

		float fAngleBase;
		if (fdA >= 0.0f) {
			// going up from angles[0]
			fAngleBase = angles[0];
			ASTRA_DEBUG("Second angle >= first angle, so assuming angles are incrementing");
		} else {
			// going down from angles[0]
			fAngleBase = angles[dims.iProjAngles - 1];
			ASTRA_DEBUG("Second angle < first angle, so assuming angles are decrementing");
		}

		// We pick the lowest end of the range, and then
		// move all angles so they fall in [0,2pi)

		std::vector<float> fRelAngles(dims.iProjAngles);
		for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
			float f = angles[i] - fAngleBase;
			while (f >= 2*M_PI)
				f -= 2*M_PI;
			while (f < 0)
				f += 2*M_PI;
			fRelAngles[i] = f;
		}

		float fRange = fabs(fRelAngles[dims.iProjAngles-1] - fRelAngles[0]);
		// Adjust for discretisation
		fRange /= dims.iProjAngles - 1;
		fRange *= dims.iProjAngles;

		ASTRA_DEBUG("Assuming angles are linearly ordered and equally spaced for Parker weighting. Angular range %f radians", fRange);
		float fScale = fRange / M_PI;

		bool ok = true;
		ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_angle, &fRelAngles[0],
		                                        dims.iProjAngles*sizeof(float), 0,
		                                        cudaMemcpyHostToDevice, stream),
		                "FBP_PreWeight transfer");
		ok &= checkCuda(cudaStreamSynchronize(stream), "FBP_PreWeight");
		if (!ok) {
			logCuda(cudaStreamDestroy(stream), "FBP_PreWeight destroy stream");
			return false;
		}

		float fCentralFanAngle = fabs(atanf(fDetUSize * (dims.iProjU*0.5f) /
		                               (fSrcOrigin + fDetOrigin)));

		if (fRange + 1e-3 < M_PI + 2*fCentralFanAngle) {
			ASTRA_WARN("Angular range (%f rad) smaller than Parker weighting range (%f rad)", fRange, M_PI + 2*fCentralFanAngle);
		}

		devFBP_ParkerWeight<<<dimGrid, dimBlock, 0, stream>>>(D_projData.ptr, projPitch, 0, dims.iProjAngles, fSrcOrigin, fDetOrigin, fDetUSize, fCentralFanAngle, fScale, dims);
	}
	if (!checkCuda(cudaStreamSynchronize(stream), "FBP_PreWeight")) {
		logCuda(cudaStreamDestroy(stream), "FBP_PreWeight destroy stream");
		return false;
	}

	logCuda(cudaStreamDestroy(stream), "FBP_PreWeight destroy stream");

	return true;
}


// ------------- end of code borrowed from FDK -------------------

bool FBP(astra::CData2D *D_vol, astra::CData2D *D_proj,
         const astra::Geometry2DParameters &geometry, SProjectorParams2D params,
	 const SFilter_internal *f,
	 bool shortScan)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(D_proj->getStorage());
	assert(projs);

	const astraCUDA::CDataGPU *vols = dynamic_cast<const astraCUDA::CDataGPU*>(D_vol->getStorage());
	assert(vols);

	assert(!projs->getArray());
	assert(!vols->getArray());

	float *D_sinoData = (float *)projs->getPtr().ptr;
	int sinoPitch = projs->getPtr().pitch / sizeof(float);
	float *D_volumeData = (float *)vols->getPtr().ptr;
	int volumePitch = vols->getPtr().pitch / sizeof(float);




	const SDimensions &dims = geometry.getDims();

	zeroGPUMemory(D_vol);

	bool ok = false;

	float fFanDetSize = 0.0f;
	if (geometry.isFan()) {
		// Call FDK_PreWeight to handle fan beam geometry. We treat
		// this as a cone beam setup of a single slice:

		// TODO: TOffsets affects this preweighting...

		// TODO: We take the fan parameters from the last projection here
		// without checking if they're the same in all projections

		std::vector<float> pfAngles;
		pfAngles.resize(dims.iProjAngles);

		float fOriginSource, fOriginDetector, fOffset;
		for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
			bool ok = astra::getFanParameters(geometry.getFan()[i], dims.iProjDets,
			                                  pfAngles[i],
			                                  fOriginSource, fOriginDetector,
			                                  fFanDetSize, fOffset);
			if (!ok) {
				ASTRA_ERROR("FBP_CUDA: Failed to extract circular fan beam parameters from fan beam geometry");
				return false;
			}
		}

		// We create a fake cudaPitchedPtr
		cudaPitchedPtr tmp;
		tmp.ptr = D_sinoData;
		tmp.pitch = sinoPitch * sizeof(float);
		tmp.xsize = dims.iProjDets;
		tmp.ysize = dims.iProjAngles;
		// and a fake Dimensions3D
		astraCUDA3d::SDimensions3D dims3d;
		dims3d.iVolX = dims.iVolWidth;
		dims3d.iVolY = dims.iVolHeight;
		dims3d.iVolZ = 1;
		dims3d.iProjAngles = dims.iProjAngles;
		dims3d.iProjU = dims.iProjDets;
		dims3d.iProjV = 1;

		FBP_PreWeight(tmp, fOriginSource,
		              fOriginDetector, 0.0f,
		              fFanDetSize, 1.0f,
		              shortScan, dims3d, &pfAngles[0]);
	} else {
		// TODO: How should different detector pixel size in different
		// projections be handled?
	}

	if (f->D_filter) {

		int iPaddedSize = astra::calcNextPowerOfTwo(2 * dims.iProjDets);
		int iFourierSize = astra::calcFFTFourierSize(iPaddedSize);

		cufftComplex * D_pcFourier = NULL;

		allocateComplexOnDevice(dims.iProjAngles, iFourierSize, &D_pcFourier);

		runCudaFFT(dims.iProjAngles, D_sinoData, sinoPitch, dims.iProjDets, iPaddedSize, D_pcFourier);

		applyFilter(dims.iProjAngles, iFourierSize, D_pcFourier, f->D_filter, f->single);

		runCudaIFFT(dims.iProjAngles, D_pcFourier, D_sinoData, sinoPitch, dims.iProjDets, iPaddedSize);

		freeComplexOnDevice(D_pcFourier);

	}

	if (geometry.isFan()) {
		ok = FanBP_FBPWeighted(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, params, geometry.getFan());

	} else {
		// scale by number of angles. For the fan-beam case, this is already
		// handled by FDK_PreWeight
		params.fOutputScale *= (M_PI / 2.0f) / (float)dims.iProjAngles;


		ok = BP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, params, geometry.getParallel());
	}
	if(!ok)
	{
		return false;
	}

	return true;
}

}
