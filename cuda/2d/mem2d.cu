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

#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/mem2d.h"
#include "astra/cuda/2d/fan_fp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/par_fp.h"
#include "astra/cuda/2d/par_bp.h"

#include "astra/cuda/2d/astra.h"

#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Logging.h"

#include "astra/GeometryUtil2D.h"
#include "astra/Data2D.h"

#include <cstdio>
#include <cassert>



namespace astraCUDA {


astra::CDataStorage *allocateGPUMemory(unsigned int x, unsigned int y, Mem3DZeroMode zero)
{
	cudaPitchedPtr ptr;

	size_t free = availableGPUMemory();

	size_t raw_pitch;
	void* raw_ptr;

	if (!checkCuda(cudaMallocPitch(&raw_ptr, &raw_pitch, sizeof(float)*x, y), "allocateGPUMemory mallocpitch")) {
		return nullptr;
	}

	ptr = make_cudaPitchedPtr(raw_ptr, raw_pitch, x, y);

	size_t free2 = availableGPUMemory();

	ASTRA_DEBUG("Allocated %d x %d on GPU. (Pre: %lu, post: %lu)", x, y, free, free2);



	if (zero == INIT_ZERO) {
		if (!checkCuda(cudaMemset2D(ptr.ptr, ptr.pitch, 0, sizeof(float)*x, y), "allocateGPUMemory memset2d")) {
			logCuda(cudaFree(ptr.ptr), "allocateGPUMemory free");
			return nullptr;
		}
	}

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(ptr);

	return ret;
}

astra::CDataStorage *allocateGPUMemoryLike(const astra::CData2D *model, Mem3DZeroMode zero)
{
	assert(model);

	std::array<int, 2> dims = model->getShape();

	return allocateGPUMemory(dims[0], dims[1], zero);
}

astra::CFloat32VolumeData2D *createGPUVolumeData2DLike(const astra::CFloat32VolumeData2D *model)
{
	astra::CDataStorage *storage = allocateGPUMemoryLike(model, INIT_ZERO);
	if (!storage)
		return nullptr;

	return new astra::CFloat32VolumeData2D(model->getGeometry(), storage);
}


astra::CFloat32ProjectionData2D *createGPUProjectionData2DLike(const astra::CFloat32ProjectionData2D *model)
{
	astra::CDataStorage *storage = allocateGPUMemoryLike(model, INIT_ZERO);
	if (!storage)
		return nullptr;

	return new astra::CFloat32ProjectionData2D(model->getGeometry(), storage);
}

astra::CData2D *createGPUData2DLike(const astra::CData2D *model)
{
	assert(model->getStorage());
	assert(model->getStorage()->isFloat32());

	std::array<int, 2> dims = model->getShape();
	astra::CDataStorage *storage = allocateGPUMemory(dims[0], dims[1], INIT_ZERO);
	if (!storage)
		return nullptr;
	return new astra::CData2D(dims[0], dims[1], storage);
}



bool zeroGPUMemory(astra::CData2D *data)
{
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(ds);

	std::array<int, 2> dims = data->getShape();

	assert(!ds->getArray());
	return checkCuda(cudaMemset2D(ds->getPtr().ptr, ds->getPtr().pitch, 0, sizeof(float)*dims[0], dims[1]), "zeroGPUMemory memset2d");
}

bool freeGPUMemory(astra::CData2D *data)
{
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(ds);

	size_t free = astraCUDA::availableGPUMemory();
	assert(!ds->getArray());

	bool ok = checkCuda(cudaFree(ds->getPtr().ptr), "freeGPUMemory");
	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Freeing memory. (Pre: %lu, post: %lu)", free, free2);

	return ok;
}

bool assignGPUMemory(astra::CData2D *dst, const astra::CData2D *src)
{
	assert(dst->getShape() == src->getShape());

	astraCUDA::CDataGPU *dsts = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(dsts);
	assert(!dsts->getArray());
	const astraCUDA::CDataGPU *srcs = dynamic_cast<const astraCUDA::CDataGPU*>(src->getStorage());
	assert(srcs);
	assert(!srcs->getArray());

	std::array<int, 2> dims = src->getShape();

	return checkCuda(cudaMemcpy2D(dsts->getPtr().ptr, dsts->getPtr().pitch, srcs->getPtr().ptr, srcs->getPtr().pitch, sizeof(float)*dims[0], dims[1], cudaMemcpyDeviceToDevice), "assignGPUMemory memcpy2d");
}

bool copyToGPUMemory(const astra::CData2D *src, astra::CData2D *dst)
{
	assert(src->getShape() == dst->getShape());

	std::array<int, 2> dims = src->getShape();
	ASTRA_DEBUG("Copying %d x %d to GPU", dims[0], dims[1]);

	assert(src->isFloat32Memory());
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(ds);

	assert(!ds->getArray());

	return checkCuda(cudaMemcpy2D(ds->getPtr().ptr, ds->getPtr().pitch, src->getFloat32Memory(), sizeof(float)*dims[0], sizeof(float)*dims[0], dims[1], cudaMemcpyHostToDevice), "copyToGPUMemory memcpy2d");
}

bool copyFromGPUMemory(astra::CData2D *dst, const astra::CData2D *src)
{
	assert(src->getShape() == dst->getShape());

	std::array<int, 2> dims = src->getShape();
	ASTRA_DEBUG("Copying %d x %d from GPU", dims[0], dims[1]);

	assert(dst->isFloat32Memory());
	const astraCUDA::CDataGPU *srcs = dynamic_cast<const astraCUDA::CDataGPU*>(src->getStorage());
	assert(srcs);

	assert(!srcs->getArray());

	return checkCuda(cudaMemcpy2D(dst->getFloat32Memory(), sizeof(float)*dims[0], srcs->getPtr().ptr, srcs->getPtr().pitch, sizeof(float)*dims[0], dims[1], cudaMemcpyDeviceToHost), "copyFromGPUMemory memcpy2d");

}


bool FP(astra::CFloat32ProjectionData2D *projData, const astra::CFloat32VolumeData2D *volData, int iDetectorSuperSampling, astra::Cuda2DProjectionKernel projKernel)
{
	const astra::CProjectionGeometry2D &pProjGeom = projData->getGeometry();
	const astra::CVolumeGeometry2D &pVolGeom = volData->getGeometry();

	SProjectorParams2D params;
	assert(projKernel == astra::ker2d_default);

	params.iRaysPerDet = iDetectorSuperSampling;
	if (iDetectorSuperSampling == 0)
		return false;

	astra::Geometry2DParameters geom = astra::convertAstraGeometry(&pVolGeom, &pProjGeom);
	params.fOutputScale = geom.getOutputScale();

	return FP(projData, volData, geom, params);
}

bool FP(astra::CData2D *projData, const astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);

	const astraCUDA::CDataGPU *vols = dynamic_cast<const astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);

	assert(!projs->getArray());
	assert(!vols->getArray());

	std::array<int, 2> projDims = projData->getShape();
	std::array<int, 2> volDims = volData->getShape();
	SDimensions dims = geom.getDims();
	assert(projDims[0] == dims.iProjDets);
	assert(projDims[1] == dims.iProjAngles);
	assert(volDims[0] == dims.iVolWidth);
	assert(volDims[1] == dims.iVolHeight);

	bool ok = true;

	if (geom.isParallel()) {
		const SParProjection* pParProjs = geom.getParallel();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= FP((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), geom.getDims(), params, pParProjs);
	} else if (geom.isFan()) {
		const SFanProjection* pFanProjs = geom.getFan();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= FanFP((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), geom.getDims(), params, pFanProjs);
	} else
		ok = false;

	return ok;
}

bool BP(const astra::CFloat32ProjectionData2D *projData, astra::CFloat32VolumeData2D *volData, int iVoxelSuperSampling, astra::Cuda2DProjectionKernel projKernel)
{
	const astra::CProjectionGeometry2D &pProjGeom = projData->getGeometry();
	const astra::CVolumeGeometry2D &pVolGeom = volData->getGeometry();

	SProjectorParams2D params;
	assert(projKernel == astra::ker2d_default);

	params.iRaysPerPixelDim = iVoxelSuperSampling;

	astra::Geometry2DParameters geom = astra::convertAstraGeometry(&pVolGeom, &pProjGeom);
	params.fOutputScale = geom.getOutputScale();

	return BP(projData, volData, geom, params);
}

bool BP(const astra::CData2D *projData, astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params)
{
	const astraCUDA::CDataGPU *projs = dynamic_cast<const astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);
	assert(!projs->getArray());
	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);
	assert(!vols->getArray());

	std::array<int, 2> projDims = projData->getShape();
	std::array<int, 2> volDims = volData->getShape();
	SDimensions dims = geom.getDims();
	assert(projDims[0] == dims.iProjDets);
	assert(projDims[1] == dims.iProjAngles);
	assert(volDims[0] == dims.iVolWidth);
	assert(volDims[1] == dims.iVolHeight);

	bool ok = true;

	if (geom.isParallel()) {
		const SParProjection* pParProjs = geom.getParallel();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= BP((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), geom.getDims(), params, pParProjs);
	} else if (geom.isFan()) {
		const SFanProjection* pFanProjs = geom.getFan();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= FanBP((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), geom.getDims(), params, pFanProjs);
	} else
		ok = false;

	return ok;

}


_AstraExport astra::CDataStorage *wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int pitch)
{
	cudaPitchedPtr ptr;
	ptr.ptr = D_ptr;
	ptr.xsize = sizeof(float) * x;
	ptr.pitch = sizeof(float) * pitch;
	ptr.ysize = y;

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(ptr);

	return ret;
}


bool FP_SART(astra::CData2D *projData, const astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params, int angle)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);

	const astraCUDA::CDataGPU *vols = dynamic_cast<const astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);

	assert(!projs->getArray());
	assert(!vols->getArray());

	std::array<int, 2> projDims = projData->getShape();
	std::array<int, 2> volDims = volData->getShape();
	SDimensions dims = geom.getDims();
	assert(projDims[0] == dims.iProjDets);
	assert(projDims[1] == 1);
	assert(volDims[0] == dims.iVolWidth);
	assert(volDims[1] == dims.iVolHeight);

	dims.iProjAngles = 1; // single angle for SART

	bool ok = true;

	if (geom.isParallel()) {
		const SParProjection* pParProjs = geom.getParallel();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= FP((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), dims, params, &pParProjs[angle]);
	} else if (geom.isFan()) {
		const SFanProjection* pFanProjs = geom.getFan();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= FanFP((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), dims, params, &pFanProjs[angle]);
	} else
		ok = false;

	return ok;
}

bool BP_SART(const astra::CData2D *projData, astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params, int angle)
{
	const astraCUDA::CDataGPU *projs = dynamic_cast<const astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);
	assert(!projs->getArray());
	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);
	assert(!vols->getArray());

	std::array<int, 2> projDims = projData->getShape();
	std::array<int, 2> volDims = volData->getShape();
	SDimensions dims = geom.getDims();
	assert(projDims[0] == dims.iProjDets);
	assert(projDims[1] == 1);
	assert(volDims[0] == dims.iVolWidth);
	assert(volDims[1] == dims.iVolHeight);

	bool ok = true;

	if (geom.isParallel()) {
		const SParProjection* pParProjs = geom.getParallel();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= BP_SART((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), angle, dims, params, pParProjs);
	} else if (geom.isFan()) {
		const SFanProjection* pFanProjs = geom.getFan();

		//switch (params.projKernel) {
		//case ker2d_default:
		ok &= FanBP_SART((float*)vols->getPtr().ptr, vols->getPtr().pitch / sizeof(float), (float*)projs->getPtr().ptr, projs->getPtr().pitch / sizeof(float), angle, dims, params, pFanProjs);
	} else
		ok = false;

	return ok;

}

bool dotProduct2D(const astra::CData2D *D_data, float &fRet)
{
	const astraCUDA::CDataGPU *datas = dynamic_cast<const astraCUDA::CDataGPU*>(D_data->getStorage());
	assert(datas);
	assert(!datas->getArray());

	std::array<int, 2> dims = D_data->getShape();

	return dotProduct2D((float*)datas->getPtr().ptr, datas->getPtr().pitch/sizeof(float), dims[0], dims[1], fRet);
}




}
