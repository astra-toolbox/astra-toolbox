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

	return std::min(props.maxTexture3D[0], std::min(props.maxTexture3D[1], props.maxTexture3D[2]));
}

astra::CDataStorage *allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero)
{
	cudaPitchedPtr ptr;

	size_t free = astraCUDA::availableGPUMemory();

	if (!checkCuda(cudaMalloc3D(&ptr, make_cudaExtent(sizeof(float)*x, y, z)), "allocateGPUMemory malloc3d")) {
		return nullptr;
	}

	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Allocated %d x %d x %d on GPU. (Pre: %lu, post: %lu)", x, y, z, free, free2);



	if (zero == INIT_ZERO) {
		if (!checkCuda(cudaMemset3D(ptr, 0, make_cudaExtent(sizeof(float)*x, y, z)), "allocateGPUMemory memset3d")) {
			cudaFree(ptr.ptr);
			return nullptr;
		}
	}

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(ptr);

	return ret;
}

bool zeroGPUMemory(astra::CData3D *data, unsigned int x, unsigned int y, unsigned int z)
{
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(ds);

	assert(!ds->getArray());
	return checkCuda(cudaMemset3D(ds->getPtr(), 0, make_cudaExtent(sizeof(float)*x, y, z)), "zeroGPUMemory");
}

bool freeGPUMemory(astra::CData3D *data)
{
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(ds);

	size_t free = astraCUDA::availableGPUMemory();
	bool ok;
	if (ds->getArray())
		ok = checkCuda(cudaFreeArray(ds->getArray()), "freeGPUMemory array");
	else
		ok = checkCuda(cudaFree(ds->getPtr().ptr), "freeGPUMemory");
	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Freeing memory. (Pre: %lu, post: %lu)", free, free2);

	return ok;
}

bool copyToGPUMemory(const astra::CData3D *src, astra::CData3D *dst, const SSubDimensions3D &pos)
{
	ASTRA_DEBUG("Copying %d x %d x %d to GPU", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);

	assert(src->isFloat32Memory());
	astraCUDA::CDataGPU *ds = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(ds);

	assert(!ds->getArray());

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

	if (!ss->getArray()) {
		p.srcArray = 0;
		p.srcPtr = ss->getPtr();
		p.extent = make_cudaExtent(pos.subnx * sizeof(float), pos.subny, pos.subnz);
	} else {
		// CUDA interface limitation: cast const away
		p.srcArray = const_cast<cudaArray*>(ss->getArray());
		p.srcPtr.ptr = 0;
		p.extent = make_cudaExtent(pos.subnx, pos.subny, pos.subnz);
	}

	p.kind = cudaMemcpyDeviceToHost;

	return checkCuda(cudaMemcpy3D(&p), "copyFromGPUMemory");
}


bool FP(const astra::CProjectionGeometry3D* pProjGeom, astra::CData3D *projData, const astra::CVolumeGeometry3D* pVolGeom, astra::CData3D *volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);

	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);

	assert(!projs->getArray());
	assert(!vols->getArray());
	SProjectorParams3D params;
	params.projKernel = projKernel;

	params.iRaysPerDetDim = iDetectorSuperSampling;
	if (iDetectorSuperSampling == 0)
		return false;

	auto res = astra::convertAstraGeometry(pVolGeom, pProjGeom);
	params.volScale = res.getVolScale();

	bool ok = true;

	if (res.isParallel()) {
		const SPar3DProjection* pParProjs = res.getParallel();

		switch (projKernel) {
		case ker3d_default:
		case ker3d_2d_weighting:
		case ker3d_matched_bp:
			ok &= Par3DFP(vols->getPtr(), projs->getPtr(), res.getDims(), pParProjs, params);
			break;
		case ker3d_sum_square_weights:
			ok &= Par3DFP_SumSqW(vols->getPtr(), projs->getPtr(), res.getDims(), pParProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (res.isCone()) {
		const SConeProjection* pConeProjs = res.getCone();

		switch (projKernel) {
		case ker3d_default:
		case ker3d_fdk_weighting:
		case ker3d_2d_weighting:
		case ker3d_matched_bp:
			ok &= ConeFP(vols->getPtr(), projs->getPtr(), res.getDims(), pConeProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (res.isCylCone()) {
		const SCylConeProjection* pCylConeProjs = res.getCylCone();
		switch (projKernel) {
		case ker3d_default: case ker3d_matched_bp:
			ok &= ConeCylFP(vols->getPtr(), projs->getPtr(), res.getDims(), pCylConeProjs, params);
			break;
		default:
			ok = false;
		}
	} else
		ok = false;

	return ok;
}

bool BP(const astra::CProjectionGeometry3D* pProjGeom, astra::CData3D *projData, const astra::CVolumeGeometry3D* pVolGeom, astra::CData3D *volData, int iVoxelSuperSampling, astra::Cuda3DProjectionKernel projKernel)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);

	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);

	assert(!vols->getArray());
	SProjectorParams3D params;
	params.projKernel = projKernel;

	params.iRaysPerVoxelDim = iVoxelSuperSampling;

	auto res = astra::convertAstraGeometry(pVolGeom, pProjGeom);
	params.volScale = res.getVolScale();

	bool ok = true;

	if (res.isParallel()) {
		const SPar3DProjection* pParProjs = res.getParallel();
		switch (projKernel) {
		case ker3d_default:
		case ker3d_2d_weighting:
			if (projs->getArray())
				ok &= Par3DBP_Array(vols->getPtr(), projs->getArray(), res.getDims(), pParProjs, params);
			else
				ok &= Par3DBP(vols->getPtr(), projs->getPtr(), res.getDims(), pParProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (res.isCone()) {
		const SConeProjection* pConeProjs = res.getCone();
		switch (projKernel) {
		case ker3d_default:
		case ker3d_fdk_weighting:
		case ker3d_2d_weighting:
			if (projs->getArray())
				ok &= ConeBP_Array(vols->getPtr(), projs->getArray(), res.getDims(), pConeProjs, params);
			else
				ok &= ConeBP(vols->getPtr(), projs->getPtr(), res.getDims(), pConeProjs, params);
			break;
		default:
			ok = false;
		}
	} else if (res.isCylCone()) {
		const SCylConeProjection* pCylConeProjs = res.getCylCone();
		// TODO: Add support for ker3d_2d_weighting?
		// TODO: Add support for ker3d_fdk_weighting?
		if (projKernel == ker3d_default) {
			if (projs->getArray())
				ok &= ConeCylBP_Array(vols->getPtr(), projs->getArray(), res.getDims(), pCylConeProjs, params);
			else
				ok &= ConeCylBP(vols->getPtr(), projs->getPtr(), res.getDims(), pCylConeProjs, params);
		} else if (projKernel == ker3d_matched_bp) {
			if (projs->getArray())
				ok &= ConeCylBP_Array_matched(vols->getPtr(), projs->getArray(), res.getDims(), pCylConeProjs, params);
			else
				ok &= ConeCylBP_matched(vols->getPtr(), projs->getPtr(), res.getDims(), pCylConeProjs, params);
		} else {
			ok = false;
		}
	} else
		ok = false;

	return ok;

}

bool FDK(const astra::CProjectionGeometry3D* pProjGeom, astra::CData3D *projData, const astra::CVolumeGeometry3D* pVolGeom, astra::CData3D *volData, bool bShortScan, const astra::SFilterConfig &filterConfig, float fOutputScale)
{
	astraCUDA::CDataGPU *projs = dynamic_cast<astraCUDA::CDataGPU*>(projData->getStorage());
	assert(projs);

	astraCUDA::CDataGPU *vols = dynamic_cast<astraCUDA::CDataGPU*>(volData->getStorage());
	assert(vols);



	assert(!projs->getArray());
	assert(!vols->getArray());
	SProjectorParams3D params;
	params.fOutputScale = fOutputScale;
	params.projKernel = ker3d_fdk_weighting;

	astra::Geometry3DParameters res = astra::convertAstraGeometry(pVolGeom, pProjGeom);
	params.volScale = res.getVolScale();

	if (!res.isCone())
		return false;

	const SConeProjection* pConeProjs = res.getCone();

	bool ok = FDK(vols->getPtr(), projs->getPtr(), pConeProjs, res.getDims(), params, bShortScan, filterConfig);

	return ok;



}

_AstraExport astra::CDataStorage *wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch)
{
	cudaPitchedPtr ptr;
	ptr.ptr = D_ptr;
	ptr.xsize = sizeof(float) * x;
	ptr.pitch = sizeof(float) * pitch;
	ptr.ysize = y;

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(ptr);

	return ret;
}

astra::CDataStorage* createProjectionArrayHandle(const float *ptr, unsigned int x, unsigned int y, unsigned int z)
{
	SDimensions3D dims;
	dims.iProjU = x;
	dims.iProjAngles = y;
	dims.iProjV = z;

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferHostProjectionsToArray(ptr, cuArray, dims);

	astraCUDA::CDataGPU *ret = new astraCUDA::CDataGPU(cuArray);

	return ret;
}

bool copyIntoArray(astra::CData3D *data, astra::CData3D *subdata, const SSubDimensions3D &pos)
{
	astraCUDA::CDataGPU *datas = dynamic_cast<astraCUDA::CDataGPU*>(data->getStorage());
	assert(datas);

	astraCUDA::CDataGPU *subdatas = dynamic_cast<astraCUDA::CDataGPU*>(subdata->getStorage());
	assert(subdatas);

	assert(datas->getArray());
	assert(!subdatas->getArray());

	ASTRA_DEBUG("Copying %d x %d x %d into GPU array", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);
	ASTRA_DEBUG("Pitch %zu, xsize %zu, ysize %zu", subdatas->getPtr().pitch, subdatas->getPtr().xsize, subdatas->getPtr().ysize);

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = make_cudaPos(0, 0, 0);
	p.srcPtr = subdatas->getPtr();

	p.dstArray = datas->getArray();
	p.dstPos = make_cudaPos(pos.subx, pos.suby, pos.subz);
	p.dstPtr.ptr = 0;

	p.extent = make_cudaExtent(pos.subnx, pos.subny, pos.subnz);

	p.kind = cudaMemcpyHostToDevice;

	return checkCuda(cudaMemcpy3D(&p), "copyIntoArray");

}



}
