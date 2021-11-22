/*
-----------------------------------------------------------------------
Copyright: 2010-2021, imec Vision Lab, University of Antwerp
           2014-2021, CWI, Amsterdam

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

#include "astra/cuda/2d/util.h"

#include "astra/Logging.h"

#include <cstdio>
#include <cassert>

namespace astraCUDA3d {


cudaPitchedPtr allocateVolumeData(const SDimensions3D& dims)
{
	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPitchedPtr volData;

	if (!checkCuda(cudaMalloc3D(&volData, extentV), "allocateVolumeData 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU buffer", dims.iVolX, dims.iVolY, dims.iVolZ);
		volData.ptr = 0;
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

	if (!checkCuda(cudaMalloc3D(&projData, extentP), "allocateProjectionData 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU buffer", dims.iProjU, dims.iProjAngles, dims.iProjV);
		projData.ptr = 0;
	}

	return projData;
}
bool zeroVolumeData(cudaPitchedPtr& D_data, const SDimensions3D& dims)
{
	char* t = (char*)D_data.ptr;

	for (unsigned int z = 0; z < dims.iVolZ; ++z) {
		if (!checkCuda(cudaMemset2D(t, D_data.pitch, 0, dims.iVolX*sizeof(float), dims.iVolY), "zeroVolumeData 3D")) {
			return false;
		}
		t += D_data.pitch * dims.iVolY;
	}
	return true;
}
bool zeroProjectionData(cudaPitchedPtr& D_data, const SDimensions3D& dims)
{
	char* t = (char*)D_data.ptr;

	for (unsigned int z = 0; z < dims.iProjV; ++z) {
		if (!checkCuda(cudaMemset2D(t, D_data.pitch, 0, dims.iProjU*sizeof(float), dims.iProjAngles), "zeroProjectionData 3D")) {
			return false;
		}
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

	return checkCuda(cudaMemcpy3D(&p), "copyVolumeToDevice 3D");
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

	return checkCuda(cudaMemcpy3D(&p), "copyProjectionsToDevice 3D");
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

	return checkCuda(cudaMemcpy3D(&p), "copyVolumeFromDevice 3D");
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

	return checkCuda(cudaMemcpy3D(&p), "copyProjectionsFromDevice 3D");
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

	return checkCuda(cudaMemcpy3D(&p), "duplicateVolumeData 3D");
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

	return checkCuda(cudaMemcpy3D(&p), "duplicateProjectionData 3D");
}



// TODO: Consider using a single array of size max(proj,volume) (per dim)
//       instead of allocating a new one each time

cudaArray* allocateVolumeArray(const SDimensions3D& dims)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaExtent extentA;
	extentA.width = dims.iVolX;
	extentA.height = dims.iVolY;
	extentA.depth = dims.iVolZ;

	if (!checkCuda(cudaMalloc3DArray(&cuArray, &channelDesc, extentA), "allocateVolumeArray 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU array", dims.iVolX, dims.iVolY, dims.iVolZ);
		return 0;
	}

	return cuArray;
}
cudaArray* allocateProjectionArray(const SDimensions3D& dims)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaExtent extentA;
	extentA.width = dims.iProjU;
	extentA.height = dims.iProjAngles;
	extentA.depth = dims.iProjV;

	if (!checkCuda(cudaMalloc3DArray(&cuArray, &channelDesc, extentA), "allocateProjectionArray 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU array", dims.iProjU, dims.iProjAngles, dims.iProjV);
		return 0;
	}

	return cuArray;
}

bool transferVolumeToArray(cudaPitchedPtr D_volumeData, cudaArray* array, const SDimensions3D& dims)
{
	cudaExtent extentA;
	extentA.width = dims.iVolX;
	extentA.height = dims.iVolY;
	extentA.depth = dims.iVolZ;

	cudaMemcpy3DParms p;
	cudaPos zp = {0, 0, 0};
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_volumeData;
	p.dstArray = array;
	p.dstPtr.ptr = 0;
	p.dstPtr.pitch = 0;
	p.dstPtr.xsize = 0;
	p.dstPtr.ysize = 0;
	p.dstPos = zp;
	p.extent = extentA;
	p.kind = cudaMemcpyDeviceToDevice;

	return checkCuda(cudaMemcpy3D(&p), "transferVolumeToArray 3D");
}

bool transferProjectionsToArray(cudaPitchedPtr D_projData, cudaArray* array, const SDimensions3D& dims)
{
	cudaExtent extentA;
	extentA.width = dims.iProjU;
	extentA.height = dims.iProjAngles;
	extentA.depth = dims.iProjV;

	cudaMemcpy3DParms p;
	cudaPos zp = {0, 0, 0};
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = D_projData;
	p.dstArray = array;
	p.dstPtr.ptr = 0;
	p.dstPtr.pitch = 0;
	p.dstPtr.xsize = 0;
	p.dstPtr.ysize = 0;
	p.dstPos = zp;
	p.extent = extentA;
	p.kind = cudaMemcpyDeviceToDevice;

	return checkCuda(cudaMemcpy3D(&p), "transferProjectionsToArray 3D");
}

bool transferHostProjectionsToArray(const float *projData, cudaArray* array, const SDimensions3D& dims)
{
	cudaExtent extentA;
	extentA.width = dims.iProjU;
	extentA.height = dims.iProjAngles;
	extentA.depth = dims.iProjV;

	cudaPitchedPtr ptr;
	ptr.ptr = (void*)projData; // const cast away
	ptr.pitch = dims.iProjU*sizeof(float);
	ptr.xsize = dims.iProjU*sizeof(float);
	ptr.ysize = dims.iProjAngles;

	cudaMemcpy3DParms p;
	cudaPos zp = {0, 0, 0};
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = ptr;
	p.dstArray = array;
	p.dstPtr.ptr = 0;
	p.dstPtr.pitch = 0;
	p.dstPtr.xsize = 0;
	p.dstPtr.ysize = 0;
	p.dstPos = zp;
	p.extent = extentA;
	p.kind = cudaMemcpyHostToDevice;

	return checkCuda(cudaMemcpy3D(&p), "transferHostProjectionsToArray 3D");
}

bool createTextureObject3D(cudaArray* array, cudaTextureObject_t& texObj)
{
	cudaChannelFormatDesc channelDesc =
	    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = array;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	return checkCuda(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL), "createTextureObject3D");
}



float dotProduct3D(cudaPitchedPtr data, unsigned int x, unsigned int y,
                   unsigned int z)
{
	return astraCUDA::dotProduct2D((float*)data.ptr, data.pitch/sizeof(float), x, y*z);
}


int calcNextPowerOfTwo(int _iValue)
{
	int iOutput = 1;
	while(iOutput < _iValue)
		iOutput *= 2;
	return iOutput;
}

}
