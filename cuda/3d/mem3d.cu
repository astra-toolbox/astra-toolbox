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

#include <cstdio>
#include <cassert>

#include "util3d.h"

#include "mem3d.h"

#include "astra3d.h"
#include "cone_fp.h"
#include "cone_bp.h"
#include "par3d_fp.h"
#include "par3d_bp.h"

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

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

#if 1
	dims.iRaysPerDetDim = iDetectorSuperSampling;
	if (iDetectorSuperSampling == 0)
		return false;
#else
	dims.iRaysPerDetDim = 1;
	astra::Cuda3DProjectionKernel projKernel = astra::ker3d_default;
#endif


	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	float outputScale = 1.0f;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          outputScale);

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
			ok &= Par3DFP(volData.d->ptr, projData.d->ptr, dims, pParProjs, outputScale);
			break;
		case astra::ker3d_sum_square_weights:
			ok &= Par3DFP_SumSqW(volData.d->ptr, projData.d->ptr, dims, pParProjs, outputScale*outputScale);
			break;
		default:
			ok = false;
		}
	} else {
		switch (projKernel) {
		case astra::ker3d_default:
			ok &= ConeFP(volData.d->ptr, projData.d->ptr, dims, pConeProjs, outputScale);
			break;
		default:
			ok = false;
		}
	}

	return ok;
}

bool BP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iVoxelSuperSampling)
{
	SDimensions3D dims;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

#if 1
	dims.iRaysPerVoxelDim = iVoxelSuperSampling;
#else
	dims.iRaysPerVoxelDim = 1;
#endif

	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	float outputScale = 1.0f;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          outputScale);

	if (pParProjs)
		ok &= Par3DBP(volData.d->ptr, projData.d->ptr, dims, pParProjs, outputScale);
	else
		ok &= ConeBP(volData.d->ptr, projData.d->ptr, dims, pConeProjs, outputScale);

	return ok;

}




}
