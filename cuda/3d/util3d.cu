/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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
#include "../2d/util.h"

namespace astraCUDA3d {


cudaPitchedPtr allocateVolumeData(const SDimensions3D& dims)
{
	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPitchedPtr volData;

	cudaError err = cudaMalloc3D(&volData, extentV);
	if (err != cudaSuccess) {
		astraCUDA::reportCudaError(err);
		fprintf(stderr, "Failed to allocate %dx%dx%d GPU buffer\n", dims.iVolX, dims.iVolY, dims.iVolZ);
		volData.ptr = 0;
		// TODO: return 0 somehow?
	}

	return volData;
}
cudaPitchedPtr allocateProjectionData(const SDimensions3D& dims)
{
	cudaExtent extentP;
	extentP.width = dims.iProjU*sizeof(float);
	extentP.height = dims.iProjAngles;
	extentP.depth = dims.iProjV;

	cudaPitchedPtr projData;

	cudaError err = cudaMalloc3D(&projData, extentP);
	if (err != cudaSuccess) {
		astraCUDA::reportCudaError(err);
		fprintf(stderr, "Failed to allocate %dx%dx%d GPU buffer\n", dims.iProjU, dims.iProjAngles, dims.iProjV);
		projData.ptr = 0;
		// TODO: return 0 somehow?
	}

	return projData;
}
bool zeroVolumeData(cudaPitchedPtr& D_data, const SDimensions3D& dims)
{
	char* t = (char*)D_data.ptr;
	cudaError err;

	for (unsigned int z = 0; z < dims.iVolZ; ++z) {
		err = cudaMemset2D(t, D_data.pitch, 0, dims.iVolX*sizeof(float), dims.iVolY);
		ASTRA_CUDA_ASSERT(err);
		t += D_data.pitch * dims.iVolY;
	}
	return true;
}
bool zeroProjectionData(cudaPitchedPtr& D_data, const SDimensions3D& dims)
{
	char* t = (char*)D_data.ptr;
	cudaError err;

	for (unsigned int z = 0; z < dims.iProjV; ++z) {
		err = cudaMemset2D(t, D_data.pitch, 0, dims.iProjU*sizeof(float), dims.iProjAngles);
		ASTRA_CUDA_ASSERT(err);
		t += D_data.pitch * dims.iProjAngles;
	}

	return true;
}
bool copyVolumeToDevice(const float* data, cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch)
{
	if (!pitch)
		pitch = dims.iVolX;

	cudaPitchedPtr ptr;
	ptr.ptr = (void*)data; // const cast away
	ptr.pitch = pitch*sizeof(float);
	ptr.xsize = dims.iVolX*sizeof(float);
	ptr.ysize = dims.iVolY;

	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = ptr;
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = D_data;
	p.extent = extentV;
	p.kind = cudaMemcpyHostToDevice;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}

bool copyProjectionsToDevice(const float* data, cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch)
{
	if (!pitch)
		pitch = dims.iProjU;

	cudaPitchedPtr ptr;
	ptr.ptr = (void*)data; // const cast away
	ptr.pitch = pitch*sizeof(float);
	ptr.xsize = dims.iProjU*sizeof(float);
	ptr.ysize = dims.iProjAngles;

	cudaExtent extentV;
	extentV.width = dims.iProjU*sizeof(float);
	extentV.height = dims.iProjAngles;
	extentV.depth = dims.iProjV;

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = ptr;
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = D_data;
	p.extent = extentV;
	p.kind = cudaMemcpyHostToDevice;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}

bool copyVolumeFromDevice(float* data, const cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch)
{
	if (!pitch)
		pitch = dims.iVolX;

	cudaPitchedPtr ptr;
	ptr.ptr = data;
	ptr.pitch = pitch*sizeof(float);
	ptr.xsize = dims.iVolX*sizeof(float);
	ptr.ysize = dims.iVolY;

	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_data;
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = ptr;
	p.extent = extentV;
	p.kind = cudaMemcpyDeviceToHost;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}
bool copyProjectionsFromDevice(float* data, const cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch)
{
	if (!pitch)
		pitch = dims.iProjU;

	cudaPitchedPtr ptr;
	ptr.ptr = data;
	ptr.pitch = pitch*sizeof(float);
	ptr.xsize = dims.iProjU*sizeof(float);
	ptr.ysize = dims.iProjAngles;

	cudaExtent extentV;
	extentV.width = dims.iProjU*sizeof(float);
	extentV.height = dims.iProjAngles;
	extentV.depth = dims.iProjV;

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_data;
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = ptr;
	p.extent = extentV;
	p.kind = cudaMemcpyDeviceToHost;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}

bool duplicateVolumeData(cudaPitchedPtr& D_dst, const cudaPitchedPtr& D_src, const SDimensions3D& dims)
{
	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_src;
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = D_dst;
	p.extent = extentV;
	p.kind = cudaMemcpyDeviceToDevice;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}
bool duplicateProjectionData(cudaPitchedPtr& D_dst, const cudaPitchedPtr& D_src, const SDimensions3D& dims)
{
	cudaExtent extentV;
	extentV.width = dims.iProjU*sizeof(float);
	extentV.height = dims.iProjAngles;
	extentV.depth = dims.iProjV;

	cudaPos zp = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_src;
	p.dstArray = 0;
	p.dstPos = zp;
	p.dstPtr = D_dst;
	p.extent = extentV;
	p.kind = cudaMemcpyDeviceToDevice;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}



// TODO: Consider using a single array of size max(proj,volume) (per dim)
//       instead of allocating a new one each time

// TODO: Figure out a faster way of zeroing the padding?

cudaArray* allocateVolumeArray(const SDimensions3D& dims)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaExtent extentA;
	extentA.width = dims.iVolX+2;
	extentA.height = dims.iVolY+2;
	extentA.depth = dims.iVolZ+2;
	cudaError err = cudaMalloc3DArray(&cuArray, &channelDesc, extentA);
	if (err != cudaSuccess) {
		astraCUDA::reportCudaError(err);
		fprintf(stderr, "Failed to allocate %dx%dx%d GPU array\n", dims.iVolX, dims.iVolY, dims.iVolZ);
		return 0;
	}

	zeroVolumeArray(cuArray, dims);

	return cuArray;
}
cudaArray* allocateProjectionArray(const SDimensions3D& dims)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaExtent extentA;
	extentA.width = dims.iProjU+2;
	extentA.height = dims.iProjAngles;
	extentA.depth = dims.iProjV+2;
	cudaError err = cudaMalloc3DArray(&cuArray, &channelDesc, extentA);

	if (err != cudaSuccess) {
		astraCUDA::reportCudaError(err);
		fprintf(stderr, "Failed to allocate %dx%dx%d GPU array\n", dims.iProjU, dims.iProjAngles, dims.iProjV);
		return 0;
	}

	zeroProjectionArray(cuArray, dims);

	return cuArray;
}
bool zeroVolumeArray(cudaArray* array, const SDimensions3D& dims)
{
	cudaPitchedPtr zeroBuf;
	cudaExtent extentS;
	extentS.width = sizeof(float)*(dims.iVolX+2);
	extentS.height = dims.iVolY+2;
	extentS.depth = 1;

	cudaExtent extentA;
	extentA.width = dims.iVolX+2;
	extentA.height = dims.iVolY+2;
	extentA.depth = 1;



	cudaError err;
	err = cudaMalloc3D(&zeroBuf, extentS);
	ASTRA_CUDA_ASSERT(err);
	err = cudaMemset2D(zeroBuf.ptr, zeroBuf.pitch, 0, sizeof(float)*(dims.iVolX+2), dims.iVolY+2);
	ASTRA_CUDA_ASSERT(err);
	
	// zero array
	for (unsigned int i = 0; i < dims.iVolZ+2; ++i) {
		cudaMemcpy3DParms p;
		cudaPos zp = {0, 0, 0};
		cudaPos dp = {0, 0, i};
		p.srcArray = 0;
		p.srcPos = zp;
		p.srcPtr = zeroBuf;
		p.dstArray = array;
		p.dstPtr.ptr = 0;
		p.dstPtr.pitch = 0;
		p.dstPtr.xsize = 0;
		p.dstPtr.ysize = 0;
		p.dstPos = dp;
		p.extent = extentA;
		p.kind = cudaMemcpyDeviceToDevice;

		err = cudaMemcpy3D(&p);
		ASTRA_CUDA_ASSERT(err);
	}
	cudaFree(zeroBuf.ptr);

	// TODO: check errors

	return true;
}
bool zeroProjectionArray(cudaArray* array, const SDimensions3D& dims)
{
	cudaPitchedPtr zeroBuf;
	cudaExtent extentS;
	extentS.width = sizeof(float)*(dims.iProjU+2);
	extentS.height = dims.iProjAngles;
	extentS.depth = 1;
	cudaExtent extentA;
	extentA.width = dims.iProjU+2;
	extentA.height = dims.iProjAngles;
	extentA.depth = 1;


	cudaError err;
	err = cudaMalloc3D(&zeroBuf, extentS);
	ASTRA_CUDA_ASSERT(err);
	err = cudaMemset2D(zeroBuf.ptr, zeroBuf.pitch, 0, sizeof(float)*(dims.iProjU+2), dims.iProjAngles);
	ASTRA_CUDA_ASSERT(err);

	for (unsigned int i = 0; i < dims.iProjV+2; ++i) {
		cudaMemcpy3DParms p;
		cudaPos zp = {0, 0, 0};
		cudaPos dp = {0, 0, i};
		p.srcArray = 0;
		p.srcPos = zp;
		p.srcPtr = zeroBuf;
		p.dstArray = array;
		p.dstPtr.ptr = 0;
		p.dstPtr.pitch = 0;
		p.dstPtr.xsize = 0;
		p.dstPtr.ysize = 0;
		p.dstPos = dp;
		p.extent = extentA;
		p.kind = cudaMemcpyDeviceToDevice;

		err = cudaMemcpy3D(&p);
		ASTRA_CUDA_ASSERT(err);
	}
	cudaFree(zeroBuf.ptr);

	// TODO: check errors
	return true;
}


bool transferVolumeToArray(cudaPitchedPtr D_volumeData, cudaArray* array, const SDimensions3D& dims)
{
	cudaExtent extentA;
	extentA.width = dims.iVolX;
	extentA.height = dims.iVolY;
	extentA.depth = dims.iVolZ;

	cudaMemcpy3DParms p;
	cudaPos zp = {0, 0, 0};
	cudaPos dp = {1, 1, 1};
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_volumeData;
	p.dstArray = array;
	p.dstPtr.ptr = 0;
	p.dstPtr.pitch = 0;
	p.dstPtr.xsize = 0;
	p.dstPtr.ysize = 0;
	p.dstPos = dp;
	p.extent = extentA;
	p.kind = cudaMemcpyDeviceToDevice;

	cudaError err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);
	// TODO: check errors

	return true;
}
bool transferProjectionsToArray(cudaPitchedPtr D_projData, cudaArray* array, const SDimensions3D& dims)
{
	cudaExtent extentA;
	extentA.width = dims.iProjU;
	extentA.height = dims.iProjAngles;
	extentA.depth = dims.iProjV;

	cudaMemcpy3DParms p;
	cudaPos zp = {0, 0, 0};
	cudaPos dp = {1, 0, 1};
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_projData;
	p.dstArray = array;
	p.dstPtr.ptr = 0;
	p.dstPtr.pitch = 0;
	p.dstPtr.xsize = 0;
	p.dstPtr.ysize = 0;
	p.dstPos = dp;
	p.extent = extentA;
	p.kind = cudaMemcpyDeviceToDevice;

	cudaError err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	// TODO: check errors

	return true;
}


float dotProduct3D(cudaPitchedPtr data, unsigned int x, unsigned int y,
                   unsigned int z)
{
	return astraCUDA::dotProduct2D((float*)data.ptr, data.pitch/sizeof(float), x, y*z, 0, 0);
}


bool cudaTextForceKernelsCompletion()
{
	cudaError_t returnedCudaError = cudaThreadSynchronize();

	if(returnedCudaError != cudaSuccess) {
		fprintf(stderr, "Failed to force completion of cuda kernels: %d: %s.\n", returnedCudaError, cudaGetErrorString(returnedCudaError));
		return false;
	}

	return true;
}

int calcNextPowerOfTwo(int _iValue)
{
	int iOutput = 1;
	while(iOutput < _iValue)
		iOutput *= 2;
	return iOutput;
}

}
