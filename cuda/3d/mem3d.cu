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

#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/mem3d.h"
#include "astra/cuda/3d/astra3d.h"
#include "astra/cuda/3d/cone_fp.h"
#include "astra/cuda/3d/cone_bp.h"
#include "astra/cuda/3d/par3d_fp.h"
#include "astra/cuda/3d/par3d_bp.h"
#include "astra/cuda/3d/fdk.h"

#include "astra/cuda/2d/astra.h"

#include "astra/Logging.h"

#include <cstdio>
#include <cassert>



namespace astraCUDA3d {


struct SMemHandle3D_internal
{
	cudaPitchedPtr ptr;
	cudaArray *arr;
	unsigned int nx;
	unsigned int ny;
	unsigned int nz;
};

int maxBlockDimension()
{
	int dev;
	cudaError_t err = cudaGetDevice(&dev);
	if (err != cudaSuccess) {
		ASTRA_WARN("Error querying device");
		return 0;
	}

	cudaDeviceProp props;
	err = cudaGetDeviceProperties(&props, dev);
	if (err != cudaSuccess) {
		ASTRA_WARN("Error querying device %d properties", dev);
		return 0;
	}

	return std::min(props.maxTexture3D[0], std::min(props.maxTexture3D[1], props.maxTexture3D[2]));
}

MemHandle3D allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero)
{
	SMemHandle3D_internal hnd;
	hnd.nx = x;
	hnd.ny = y;
	hnd.nz = z;
	hnd.arr = 0;

	size_t free = astraCUDA::availableGPUMemory();

	cudaError_t err;
	err = cudaMalloc3D(&hnd.ptr, make_cudaExtent(sizeof(float)*x, y, z));

	if (err != cudaSuccess) {
		return MemHandle3D();
	}

	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Allocated %d x %d x %d on GPU. (Pre: %lu, post: %lu)", x, y, z, free, free2);



	if (zero == INIT_ZERO) {
		err = cudaMemset3D(hnd.ptr, 0, make_cudaExtent(sizeof(float)*x, y, z));
		if (err != cudaSuccess) {
			cudaFree(hnd.ptr.ptr);
			return MemHandle3D();
		}
	}

	MemHandle3D ret;
	ret.d = boost::shared_ptr<SMemHandle3D_internal>(new SMemHandle3D_internal);
	*ret.d = hnd;

	return ret;
}

bool zeroGPUMemory(MemHandle3D handle, unsigned int x, unsigned int y, unsigned int z)
{
	SMemHandle3D_internal& hnd = *handle.d.get();
	assert(!hnd.arr);
	cudaError_t err = cudaMemset3D(hnd.ptr, 0, make_cudaExtent(sizeof(float)*x, y, z));
	return err == cudaSuccess;
}

bool freeGPUMemory(MemHandle3D handle)
{
	size_t free = astraCUDA::availableGPUMemory();
	cudaError_t err;
	if (handle.d->arr)
		err = cudaFreeArray(handle.d->arr);
	else
		err = cudaFree(handle.d->ptr.ptr);
	size_t free2 = astraCUDA::availableGPUMemory();

	ASTRA_DEBUG("Freeing memory. (Pre: %lu, post: %lu)", free, free2);

	return err == cudaSuccess;
}

bool copyToGPUMemory(const float *src, MemHandle3D dst, const SSubDimensions3D &pos)
{
	ASTRA_DEBUG("Copying %d x %d x %d to GPU", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);
	assert(!dst.d->arr);
	cudaPitchedPtr s;
	s.ptr = (void*)src; // const cast away
	s.pitch = pos.pitch * sizeof(float);
	s.xsize = pos.nx * sizeof(float);
	s.ysize = pos.ny;
	ASTRA_DEBUG("Pitch %d, xsize %d, ysize %d", s.pitch, s.xsize, s.ysize);

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = make_cudaPos(pos.subx * sizeof(float), pos.suby, pos.subz);
	p.srcPtr = s;

	p.dstArray = 0;
	p.dstPos = make_cudaPos(0, 0, 0);
	p.dstPtr = dst.d->ptr;

	p.extent = make_cudaExtent(pos.subnx * sizeof(float), pos.subny, pos.subnz);

	p.kind = cudaMemcpyHostToDevice;

	cudaError_t err = cudaMemcpy3D(&p);

	return err == cudaSuccess;
}


bool copyFromGPUMemory(float *dst, MemHandle3D src, const SSubDimensions3D &pos)
{
	ASTRA_DEBUG("Copying %d x %d x %d from GPU", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);
	assert(!src.d->arr);
	cudaPitchedPtr d;
	d.ptr = (void*)dst;
	d.pitch = pos.pitch * sizeof(float);
	d.xsize = pos.nx * sizeof(float);
	d.ysize = pos.ny;
	ASTRA_DEBUG("Pitch %d, xsize %d, ysize %d", d.pitch, d.xsize, d.ysize);

	cudaMemcpy3DParms p;
	p.srcPos = make_cudaPos(0, 0, 0);

	p.dstArray = 0;
	p.dstPos = make_cudaPos(pos.subx * sizeof(float), pos.suby, pos.subz);
	p.dstPtr = d;

        if (src.d->ptr.ptr) {
            p.srcArray = 0;
            p.srcPtr = src.d->ptr;
            p.extent = make_cudaExtent(pos.subnx * sizeof(float), pos.subny, pos.subnz);
        } else {
            p.srcArray = src.d->arr;
            p.srcPtr.ptr = 0;
            p.extent = make_cudaExtent(pos.subnx, pos.subny, pos.subnz);
        }

	p.kind = cudaMemcpyDeviceToHost;

	cudaError_t err = cudaMemcpy3D(&p);

	return err == cudaSuccess;

}


bool FP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel)
{
	assert(!projData.d->arr);
	assert(!volData.d->arr);
	SDimensions3D dims;
	SProjectorParams3D params;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

#if 1
	params.iRaysPerDetDim = iDetectorSuperSampling;
	if (iDetectorSuperSampling == 0)
		return false;
#else
	astra::Cuda3DProjectionKernel projKernel = astra::ker3d_default;
#endif


	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          params);

	if (pParProjs) {
#if 0
		for (int i = 0; i < dims.iProjAngles; ++i) {
			ASTRA_DEBUG("Vec: %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n",
			    pParProjs[i].fRayX, pParProjs[i].fRayY, pParProjs[i].fRayZ,
			    pParProjs[i].fDetSX, pParProjs[i].fDetSY, pParProjs[i].fDetSZ,
			    pParProjs[i].fDetUX, pParProjs[i].fDetUY, pParProjs[i].fDetUZ,
			    pParProjs[i].fDetVX, pParProjs[i].fDetVY, pParProjs[i].fDetVZ);
		}
#endif

		switch (projKernel) {
		case astra::ker3d_default:
			ok &= Par3DFP(volData.d->ptr, projData.d->ptr, dims, pParProjs, params);
			break;
		case astra::ker3d_sum_square_weights:
			ok &= Par3DFP_SumSqW(volData.d->ptr, projData.d->ptr, dims, pParProjs, params);
			break;
		default:
			ok = false;
		}
	} else {
		switch (projKernel) {
		case astra::ker3d_default:
			ok &= ConeFP(volData.d->ptr, projData.d->ptr, dims, pConeProjs, params);
			break;
		default:
			ok = false;
		}
	}

	delete[] pParProjs;
	delete[] pConeProjs;

	return ok;
}

bool BP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iVoxelSuperSampling, bool bFDKWeighting)
{
	assert(!volData.d->arr);
	SDimensions3D dims;
	SProjectorParams3D params;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

#if 1
	params.iRaysPerVoxelDim = iVoxelSuperSampling;
#endif

	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          params);

	params.bFDKWeighting = bFDKWeighting;

	if (pParProjs) {
		if (projData.d->arr)
			ok &= Par3DBP_Array(volData.d->ptr, projData.d->arr, dims, pParProjs, params);
		else
			ok &= Par3DBP(volData.d->ptr, projData.d->ptr, dims, pParProjs, params);
	} else {
		if (projData.d->arr)
			ok &= ConeBP_Array(volData.d->ptr, projData.d->arr, dims, pConeProjs, params);
		else
			ok &= ConeBP(volData.d->ptr, projData.d->ptr, dims, pConeProjs, params);
	}

	delete[] pParProjs;
	delete[] pConeProjs;

	return ok;

}

bool FDK(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, bool bShortScan, const float *pfFilter)
{
	assert(!projData.d->arr);
	assert(!volData.d->arr);
	SDimensions3D dims;
	SProjectorParams3D params;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          params);

	if (!ok || !pConeProjs) {
		delete[] pParProjs;
		delete[] pConeProjs;
		return false;
	}

	ok &= FDK(volData.d->ptr, projData.d->ptr, pConeProjs, dims, params, bShortScan, pfFilter);

	delete[] pParProjs;
	delete[] pConeProjs;

	return ok;



}

_AstraExport MemHandle3D wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch)
{
	cudaPitchedPtr ptr;
	ptr.ptr = D_ptr;
	ptr.xsize = sizeof(float) * x;
	ptr.pitch = sizeof(float) * pitch;
	ptr.ysize = y;

	SMemHandle3D_internal h;
	h.ptr = ptr;
	h.arr = 0;

	MemHandle3D hnd;
	hnd.d = boost::shared_ptr<SMemHandle3D_internal>(new SMemHandle3D_internal);
	*hnd.d = h;

	return hnd;
}

MemHandle3D createProjectionArrayHandle(const float *ptr, unsigned int x, unsigned int y, unsigned int z)
{
	SDimensions3D dims;
	dims.iProjU = x;
	dims.iProjAngles = y;
	dims.iProjV = z;

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferHostProjectionsToArray(ptr, cuArray, dims);

	SMemHandle3D_internal h;
	h.arr = cuArray;
	h.ptr.ptr = 0;

	MemHandle3D hnd;
	hnd.d = boost::shared_ptr<SMemHandle3D_internal>(new SMemHandle3D_internal);
	*hnd.d = h;

	return hnd;
}

bool copyIntoArray(MemHandle3D handle, MemHandle3D subdata, const SSubDimensions3D &pos)
{
	assert(handle.d->arr);
	assert(!handle.d->ptr.ptr);
	assert(!subdata.d->arr);
	assert(subdata.d->ptr.ptr);

	ASTRA_DEBUG("Copying %d x %d x %d into GPU array", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);
	ASTRA_DEBUG("Pitch %d, xsize %d, ysize %d", subdata.d->ptr.pitch, subdata.d->ptr.xsize, subdata.d->ptr.ysize);

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = make_cudaPos(0, 0, 0);
	p.srcPtr = subdata.d->ptr;

	p.dstArray = handle.d->arr;
	p.dstPos = make_cudaPos(pos.subx, pos.suby, pos.subz);
	p.dstPtr.ptr = 0;

	p.extent = make_cudaExtent(pos.subnx, pos.subny, pos.subnz);

	p.kind = cudaMemcpyHostToDevice;

	cudaError_t err = cudaMemcpy3D(&p);

	return err == cudaSuccess;

}



}
