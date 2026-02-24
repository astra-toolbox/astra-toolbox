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

// For fan-beam preweighting
#include "astra/cuda/3d/fdk.h"

#include "astra/Logging.h"
#include "astra/Filters.h"
#include "astra/Data2D.h"

namespace astraCUDA {

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

		astraCUDA3d::FDK_PreWeight(tmp, fOriginSource,
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
