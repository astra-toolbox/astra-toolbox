/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
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

#include <cstdio>
#include <cassert>

#include "util3d.h"

#include "mem3d.h"

#include "astra3d.h"
#include "cone_fp.h"
#include "cone_bp.h"
#include "par3d_fp.h"
#include "par3d_bp.h"
#include "fdk.h"

#include "astra/Logging.h"


namespace astraCUDA3d {


struct SMemHandle3D_internal
{
	cudaPitchedPtr ptr;
	unsigned int nx;
	unsigned int ny;
	unsigned int nz;
};

size_t availableGPUMemory()
{
	size_t free, total;
	cudaError_t err = cudaMemGetInfo(&free, &total);
	if (err != cudaSuccess)
		return 0;
	return free;
}

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

	size_t free = availableGPUMemory();

	cudaError_t err;
	err = cudaMalloc3D(&hnd.ptr, make_cudaExtent(sizeof(float)*x, y, z));

	if (err != cudaSuccess) {
		return MemHandle3D();
	}

	size_t free2 = availableGPUMemory();

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
	cudaError_t err = cudaMemset3D(hnd.ptr, 0, make_cudaExtent(sizeof(float)*x, y, z));
	return err == cudaSuccess;
}

bool freeGPUMemory(MemHandle3D handle)
{
	size_t free = availableGPUMemory();
	cudaError_t err = cudaFree(handle.d->ptr.ptr);
	size_t free2 = availableGPUMemory();

	ASTRA_DEBUG("Freeing memory. (Pre: %lu, post: %lu)", free, free2);

	return err == cudaSuccess;
}

bool copyToGPUMemory(const float *src, MemHandle3D dst, const SSubDimensions3D &pos)
{
	ASTRA_DEBUG("Copying %d x %d x %d to GPU", pos.subnx, pos.subny, pos.subnz);
	ASTRA_DEBUG("Offset %d,%d,%d", pos.subx, pos.suby, pos.subz);
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
	cudaPitchedPtr d;
	d.ptr = (void*)dst;
	d.pitch = pos.pitch * sizeof(float);
	d.xsize = pos.nx * sizeof(float);
	d.ysize = pos.ny;
	ASTRA_DEBUG("Pitch %d, xsize %d, ysize %d", d.pitch, d.xsize, d.ysize);

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = make_cudaPos(0, 0, 0);
	p.srcPtr = src.d->ptr;

	p.dstArray = 0;
	p.dstPos = make_cudaPos(pos.subx * sizeof(float), pos.suby, pos.subz);
	p.dstPtr = d;

	p.extent = make_cudaExtent(pos.subnx * sizeof(float), pos.subny, pos.subnz);

	p.kind = cudaMemcpyDeviceToHost;

	cudaError_t err = cudaMemcpy3D(&p);

	return err == cudaSuccess;

}


bool FP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel)
{
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

	return ok;
}

bool BP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iVoxelSuperSampling, bool bFDKWeighting)
{
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

	if (pParProjs)
		ok &= Par3DBP(volData.d->ptr, projData.d->ptr, dims, pParProjs, params);
	else
		ok &= ConeBP(volData.d->ptr, projData.d->ptr, dims, pConeProjs, params);

	return ok;

}

bool FDK(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, bool bShortScan, const float *pfFilter)
{
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

	if (!ok || !pConeProjs)
		return false;

	ok &= FDK(volData.d->ptr, projData.d->ptr, pConeProjs, dims, params, bShortScan, pfFilter);

	return ok;



}

MemHandle3D wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch)
{
	cudaPitchedPtr ptr;
	ptr.ptr = D_ptr;
	ptr.xsize = sizeof(float) * x;
	ptr.pitch = sizeof(float) * pitch;
	ptr.ysize = y;

	SMemHandle3D_internal h;
	h.ptr = ptr;

	MemHandle3D hnd;
	hnd.d = boost::shared_ptr<SMemHandle3D_internal>(new SMemHandle3D_internal);
	*hnd.d = h;

	return hnd;
}



}
