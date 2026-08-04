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
#include "astra/cuda/3d/mem3d.h"
#include "astra/cuda/3d/astra3d.h"
#include "astra/cuda/3d/cone_fp.h"
#include "astra/cuda/3d/cone_bp.h"
#include "astra/cuda/3d/cone_cyl.h"
#include "astra/cuda/3d/par3d_fp.h"
#include "astra/cuda/3d/par3d_bp.h"
#include "astra/cuda/3d/fdk.h"

#include "astra/cuda/2d/astra.h"
#include "astra/cuda/2d/util.h"

#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Logging.h"
#include "astra/Filters.h"

#include "astra/Data3D.h"

#include <cstdio>
#include <cassert>



namespace astraCUDA3d {



int maxBlockDimension()
{
	int dev;
	if (!checkCuda(cudaGetDevice(&dev), "maxBlockDimension getDevice")) {
		ASTRA_WARN("Error querying device");
		return 0;
	}

	cudaDeviceProp props;
	if (!checkCuda(cudaGetDeviceProperties(&props, dev), "maxBlockDimension getDviceProps")) {
		ASTRA_WARN("Error querying device %d properties", dev);
		return 0;
	}

	int width = props.maxTexture2DLayered[0];
	int height = props.maxTexture2DLayered[1];

	return std::min(width, height);
}

std::unique_ptr<astra::CDataStorage> allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero)
{
	cudaPitchedPtr ptr;

	size_t free = astraCUDA::availableGPUMemory();

	if (!checkCuda(cudaMalloc(&ptr.ptr,(size_t)sizeof(float)*x*y*z), "allocateGPUMemory malloc")) {
		return nullptr;
	}
	ptr.pitch = ptr.xsize = sizeof(float)*x;
	ptr.ysize = y;

	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Allocated %d x %d x %d on GPU, pitch %zu. (Pre: %lu, post: %lu)", x, y, z, ptr.pitch, free, free2);



	if (zero == INIT_ZERO) {
		if (!checkCuda(cudaMemset3D(ptr, 0, make_cudaExtent(sizeof(float)*x, y, z)), "allocateGPUMemory memset3d")) {
			logCuda(cudaFree(ptr.ptr), "allocateGPUMemory free");
			return nullptr;
		}
	}

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(ptr);

	return std::unique_ptr<astra::CDataStorage>(ret);
}

std::unique_ptr<astra::CDataStorage> allocateGPUMemoryLike(const astra::CData3D *model, Mem3DZeroMode zero)
{
	assert(model);

	std::array<int, 3> dims = model->getShape();

	return allocateGPUMemory(dims[0], dims[1], dims[2], zero);
}

astra::CFloat32VolumeData3D *createGPUVolumeData3DLike(const astra::CFloat32VolumeData3D *model)
{
	std::unique_ptr<astra::CDataStorage> storage = allocateGPUMemoryLike(model, INIT_ZERO);
	if (!storage)
		return nullptr;

	return new astra::CFloat32VolumeData3D(model->getGeometry(), std::move(storage));
}


astra::CFloat32ProjectionData3D *createGPUProjectionData3DLike(const astra::CFloat32ProjectionData3D *model)
{
	std::unique_ptr<astra::CDataStorage> storage = allocateGPUMemoryLike(model, INIT_ZERO);
	if (!storage)
		return nullptr;

	return new astra::CFloat32ProjectionData3D(model->getGeometry(), std::move(storage));
}

astra::CData3D *createGPUData3DLike(const astra::CData3D *model, Mem3DZeroMode zero)
{
	assert(model->getStorage());
	assert(model->getStorage()->isFloat32());

	std::array<int, 3> dims = model->getShape();
	std::unique_ptr<astra::CDataStorage> storage = allocateGPUMemory(dims[0], dims[1], dims[2], zero);
	if (!storage)
		return nullptr;
	return new astra::CData3D(dims[0], dims[1], dims[2], std::move(storage));
}


bool zeroGPUMemory(astra::CData3D *data)
{
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(ds);

	std::array<int, 3> dims = data->getShape();

	return checkCuda(cudaMemset3D(ds->getPtr(), 0, make_cudaExtent(sizeof(float)*dims[0], dims[1], dims[2])), "zeroGPUMemory");
}

bool freeGPUMemory(astra::CData3D *data)
{
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(ds);

	size_t free = astraCUDA::availableGPUMemory();
	bool ok = checkCuda(cudaFree(ds->getPtr().ptr), "freeGPUMemory");
	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Freeing memory. (Pre: %lu, post: %lu)", free, free2);

	return ok;
}

bool assignGPUMemory(astra::CData3D *dst, const astra::CData3D *src)
{
	assert(dst->getShape() == src->getShape());

	astraCUDA::CDataGPU *dsts = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(dsts);
	const astraCUDA::CDataGPU *srcs = dynamic_cast<const astraCUDA::CDataGPU*>(src->getStorage());
	assert(srcs);

	std::array<int, 3> dims = src->getShape();

	cudaExtent extentV;
	extentV.width = dims[0]*sizeof(float);
	extentV.height = dims[1];
	extentV.depth = dims[2];

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = srcs->getPtr();
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = dsts->getPtr();
	p.extent = extentV;
	p.kind = cudaMemcpyDeviceToDevice;

	return checkCuda(cudaMemcpy3D(&p), "assignGPUMemory 3D");
}

bool copyToGPUMemory(const astra::CData3D *src, astra::CData3D *dst)
{
	assert(src->getShape() == dst->getShape());

	std::array<int, 3> dims = src->getShape();

	SSubDimensions3D pos;
	pos.nx = pos.subnx = dims[0];
	pos.ny = pos.subny = dims[1];
	pos.nz = pos.subnz = dims[2];
	pos.pitch = dims[0];
	pos.subx = pos.suby = pos.subz = 0;

	return copyToGPUMemory(src, dst, pos);
}

bool copyToGPUMemory(const astra::CData3D *src, astra::CData3D *dst, const SSubDimensions3D &pos)
{
	ASTRA_DEBUG("Copying %d x %d x %d to GPU", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);

	assert(src->isFloat32Memory());
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(ds);

	cudaPitchedPtr s;
	s.ptr = (void*)src->getFloat32Memory(); // const cast away
	s.pitch = pos.pitch * sizeof(float);
	s.xsize = pos.nx * sizeof(float);
	s.ysize = pos.ny;
	ASTRA_DEBUG("Pitch %zu, xsize %zu, ysize %zu", s.pitch, s.xsize, s.ysize);

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = make_cudaPos(pos.subx * sizeof(float), pos.suby, pos.subz);
	p.srcPtr = s;

	p.dstArray = 0;
	p.dstPos = make_cudaPos(0, 0, 0);
	p.dstPtr = ds->getPtr();

	p.extent = make_cudaExtent(pos.subnx * sizeof(float), pos.subny, pos.subnz);

	p.kind = cudaMemcpyHostToDevice;

	return checkCuda(cudaMemcpy3D(&p), "copyToGPUMemory");
}

bool copyFromGPUMemory(astra::CData3D *dst, const astra::CData3D *src)
{
	assert(src->getShape() == dst->getShape());

	std::array<int, 3> dims = dst->getShape();

	SSubDimensions3D pos;
	pos.nx = pos.subnx = dims[0];
	pos.ny = pos.subny = dims[1];
	pos.nz = pos.subnz = dims[2];
	pos.pitch = dims[0];
	pos.subx = pos.suby = pos.subz = 0;

	return copyFromGPUMemory(dst, src, pos);
}



bool copyFromGPUMemory(astra::CData3D *dst, const astra::CData3D *src, const SSubDimensions3D &pos)
{
	ASTRA_DEBUG("Copying %d x %d x %d from GPU", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);

	assert(dst->isFloat32Memory());
	const astraCUDA::CDataGPU *ss = dynamic_cast<const astraCUDA::CDataGPU*>(src->getStorage());
	assert(ss);

	//assert(!src.d->arr);
	cudaPitchedPtr d;
	d.ptr = (void*)dst->getFloat32Memory();
	d.pitch = pos.pitch * sizeof(float);
	d.xsize = pos.nx * sizeof(float);
	d.ysize = pos.ny;
	ASTRA_DEBUG("Pitch %zu, xsize %zu, ysize %zu", d.pitch, d.xsize, d.ysize);

	cudaMemcpy3DParms p;
	p.srcPos = make_cudaPos(0, 0, 0);

	p.dstArray = 0;
	p.dstPos = make_cudaPos(pos.subx * sizeof(float), pos.suby, pos.subz);
	p.dstPtr = d;

	p.srcArray = 0;
	p.srcPtr = ss->getPtr();
	p.extent = make_cudaExtent(pos.subnx * sizeof(float), pos.subny, pos.subnz);

	p.kind = cudaMemcpyDeviceToHost;

	return checkCuda(cudaMemcpy3D(&p), "copyFromGPUMemory");
}


bool FP(astra::CFloat32ProjectionData3D *projData, const astra::CFloat32VolumeData3D *volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel)
{
	const astra::CProjectionGeometry3D &pProjGeom = projData->getGeometry();
	const astra::CVolumeGeometry3D &pVolGeom = volData->getGeometry();

	SProjectorParams3D params;
	params.projKernel = projKernel;

	params.iRaysPerDetDim = iDetectorSuperSampling;
	if (iDetectorSuperSampling == 0)
		return false;

	astra::Geometry3DParameters geom = astra::convertAstraGeometry(&pVolGeom, &pProjGeom);
	params.volScale = geom.getVolScale();

	return FP(projData, volData, geom, params);
}

bool FP(astra::CData3D *projData, const astra::CData3D *volData, const astra::Geometry3DParameters &geom, SProjectorParams3D params)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);

	const astraCUDA::CDataGPU *vols = dynamic_cast<const astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);

	std::array<int, 3> projDims = projData->getShape();
	std::array<int, 3> volDims = volData->getShape();
	SDimensions3D dims = geom.getDims();
	assert(projDims[0] == dims.iProjU);
	assert(projDims[1] == dims.iProjAngles);
	assert(projDims[2] == dims.iProjV);
	assert(volDims[0] == dims.iVolX);
	assert(volDims[1] == dims.iVolY);
	assert(volDims[2] == dims.iVolZ);

	bool ok = true;

	if (geom.isParallel()) {
		const std::vector<SPar3DProjection>& pParProjs = geom.getParallel();

		switch (params.projKernel) {
		case ker3d_default:
		case ker3d_2d_weighting:
		case ker3d_matched_bp:
		case ker3d_dfdx: case ker3d_dfdy: case ker3d_dfdz:
		case ker3d_dfdxs: case ker3d_dfdys: case ker3d_dfdzs:
			ok &= Par3DFP(vols->getPtr(), projs->getPtr(), geom.getDims(), pParProjs, params);
			break;
		case ker3d_sum_square_weights:
			ok &= Par3DFP_SumSqW(projs->getPtr(), geom.getDims(), pParProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (geom.isCone()) {
		const std::vector<SConeProjection>& pConeProjs = geom.getCone();

		switch (params.projKernel) {
		case ker3d_default:
		case ker3d_fdk_weighting:
		case ker3d_2d_weighting:
		case ker3d_matched_bp:
		case ker3d_dfdx: case ker3d_dfdy: case ker3d_dfdz:
		case ker3d_dfdxs: case ker3d_dfdys: case ker3d_dfdzs:
			ok &= ConeFP(vols->getPtr(), projs->getPtr(), geom.getDims(), pConeProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (geom.isCylCone()) {
		const std::vector<SCylConeProjection>& pCylConeProjs = geom.getCylCone();
		switch (params.projKernel) {
		case ker3d_default: case ker3d_matched_bp:
			ok &= ConeCylFP(vols->getPtr(), projs->getPtr(), geom.getDims(), pCylConeProjs, params);
			break;
		default:
			ok = false;
		}
	} else
		ok = false;

	return ok;
}

bool BP(const astra::CFloat32ProjectionData3D *projData, astra::CFloat32VolumeData3D *volData, int iVoxelSuperSampling, astra::Cuda3DProjectionKernel projKernel)
{
	const astra::CProjectionGeometry3D &pProjGeom = projData->getGeometry();
	const astra::CVolumeGeometry3D &pVolGeom = volData->getGeometry();

	SProjectorParams3D params;
	params.projKernel = projKernel;

	params.iRaysPerVoxelDim = iVoxelSuperSampling;

	astra::Geometry3DParameters geom = astra::convertAstraGeometry(&pVolGeom, &pProjGeom);
	params.volScale = geom.getVolScale();

	return BP(projData, volData, geom, params);
}

bool BP(const astra::CData3D *projData, astra::CData3D *volData, const astra::Geometry3DParameters &geom, SProjectorParams3D params)
{
	const astraCUDA::CDataGPU *projs = dynamic_cast<const astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);
	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);

	std::array<int, 3> projDims = projData->getShape();
	std::array<int, 3> volDims = volData->getShape();
	SDimensions3D dims = geom.getDims();
	assert(projDims[0] == dims.iProjU);
	assert(projDims[1] == dims.iProjAngles);
	assert(projDims[2] == dims.iProjV);
	assert(volDims[0] == dims.iVolX);
	assert(volDims[1] == dims.iVolY);
	assert(volDims[2] == dims.iVolZ);

	bool ok = true;

	if (geom.isParallel()) {
		const std::vector<SPar3DProjection>& pParProjs = geom.getParallel();
		switch (params.projKernel) {
		case ker3d_default:
		case ker3d_2d_weighting:
			ok &= Par3DBP(vols->getPtr(), projs->getPtr(), geom.getDims(), pParProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (geom.isCone()) {
		const std::vector<SConeProjection>& pConeProjs = geom.getCone();
		switch (params.projKernel) {
		case ker3d_default:
		case ker3d_fdk_weighting:
		case ker3d_2d_weighting:
			ok &= ConeBP(vols->getPtr(), projs->getPtr(), geom.getDims(), pConeProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (geom.isCylCone()) {
		const std::vector<SCylConeProjection>& pCylConeProjs = geom.getCylCone();
		// TODO: Add support for ker3d_2d_weighting?
		// TODO: Add support for ker3d_fdk_weighting?
		if (params.projKernel == ker3d_default) {
			ok &= ConeCylBP(vols->getPtr(), projs->getPtr(), geom.getDims(), pCylConeProjs, params);
		} else if (params.projKernel == ker3d_matched_bp) {
			ok &= ConeCylBP_matched(vols->getPtr(), projs->getPtr(), geom.getDims(), pCylConeProjs, params);
		} else {
			ok = false;
		}
	} else
		ok = false;

	return ok;

}

bool FDK(astra::CFloat32ProjectionData3D *projData, astra::CFloat32VolumeData3D *volData, bool bShortScan, const astra::SFilterConfig &filterConfig, float fOutputScale)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);
	const astra::CProjectionGeometry3D &pProjGeom = projData->getGeometry();

	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);
	const astra::CVolumeGeometry3D &pVolGeom = volData->getGeometry();

	SProjectorParams3D params;
	params.fOutputScale = fOutputScale;
	params.projKernel = ker3d_fdk_weighting;

	astra::Geometry3DParameters res = astra::convertAstraGeometry(&pVolGeom, &pProjGeom);
	params.volScale = res.getVolScale();

	if (!res.isCone())
		return false;

	bool ok = FDK(vols->getPtr(), projs->getPtr(), res.getCone(), res.getDims(), params, bShortScan, filterConfig);

	return ok;



}

_AstraExport std::unique_ptr<astra::CDataStorage> wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch)
{
	cudaPitchedPtr ptr;
	ptr.ptr = D_ptr;
	ptr.xsize = sizeof(float) * x;
	ptr.pitch = sizeof(float) * pitch;
	ptr.ysize = y;

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(ptr);

	return std::unique_ptr<astra::CDataStorage>(ret);
}

bool dotProduct3D(const astra::CData3D *D_data, float &fRet)
{
	const astraCUDA::CDataGPU *datas = dynamic_cast<const astraCUDA::CDataGPU*>(D_data->getStorage());
	assert(datas);

	std::array<int, 3> dims = D_data->getShape();

	return astraCUDA::dotProduct2D((float*)datas->getPtr().ptr, datas->getPtr().pitch/sizeof(float), dims[0], dims[1]*dims[2], fRet);
}



}
