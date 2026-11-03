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

#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Data3D.h"
#include "astra/Logging.h"

#include <cstdio>
#include <cassert>
#include <cstring>

namespace astraCUDA3d {


static const unsigned int g_permuteBlockSlices = 4;
static const unsigned int g_permuteBlock1 = 32;
static const unsigned int g_permuteBlock2 = 32;



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

cudaPitchedPtr allocateProjectionData_UVAngle(const SDimensions3D& dims)
{
	cudaExtent extentP;
	extentP.width = dims.iProjU*sizeof(float);
	extentP.height = dims.iProjV;
	extentP.depth = dims.iProjAngles;

	cudaPitchedPtr projData;

	if (!checkCuda(cudaMalloc3D(&projData, extentP), "allocateProjectionData_UVAngle 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU buffer", dims.iProjU, dims.iProjV, dims.iProjAngles);
		projData.ptr = 0;
	}

	return projData;
}

bool zeroVolumeData(cudaPitchedPtr D_data, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	if (!checkCuda(cudaMemset3DAsync(D_data, 0, extentV, stream()), "zeroVolumeData memset3d"))
		return false;

	if (!stream.syncIfSync("zeroVolumeData sync"))
		return false;

	return true;
}

bool zeroVolumeData(astra::CData3D *D_data)
{
	astraCUDA::CDataGPU *datas = dynamic_cast<astraCUDA::CDataGPU*>(D_data->getStorage());
	assert(datas);

	std::array<int, 3> dims = D_data->getShape();

	char* t = (char*)datas->getPtr().ptr;

	for (unsigned int z = 0; z < dims[2]; ++z) {
		if (!checkCuda(cudaMemset2D(t, datas->getPtr().pitch, 0, dims[0]*sizeof(float), dims[1]), "zeroVolumeData 3D")) {
			return false;
		}
		t += datas->getPtr().pitch * dims[1];
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

bool duplicateVolumeData(astra::CData3D *D_dst, const astra::CData3D *D_src)
{
	assert(D_dst->getShape() == D_src->getShape());
	astraCUDA::CDataGPU *dsts = dynamic_cast<astraCUDA::CDataGPU*>(D_dst->getStorage());
	assert(dsts);
	const astraCUDA::CDataGPU *srcs = dynamic_cast<const astraCUDA::CDataGPU*>(D_src->getStorage());
	assert(srcs);

	std::array<int, 3> dims = D_src->getShape();


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

	return checkCuda(cudaMemcpy3D(&p), "duplicateVolumeData 3D");
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

	if (!checkCuda(cudaMalloc3DArray(&cuArray, &channelDesc, extentA, 0), "allocateVolumeArray 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU array", dims.iVolX, dims.iVolY, dims.iVolZ);
		return 0;
	}

	return cuArray;
}

cudaArray* allocateVolumeArray_LayeredZ(const SDimensions3D& dims)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaExtent extentA;
	extentA.width = dims.iVolX;
	extentA.height = dims.iVolY;
	extentA.depth = dims.iVolZ;

	if (!checkCuda(cudaMalloc3DArray(&cuArray, &channelDesc, extentA, cudaArrayLayered), "allocateVolumeArray_LayeredZ 3D")) {
		ASTRA_ERROR("Failed to allocate layered %dx%dx%d GPU array", dims.iVolX, dims.iVolY, dims.iVolZ);
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

	if (!checkCuda(cudaMalloc3DArray(&cuArray, &channelDesc, extentA, 0), "allocateProjectionArray 3D")) {
		ASTRA_ERROR("Failed to allocate %dx%dx%d GPU array", dims.iProjU, dims.iProjAngles, dims.iProjV);
		return 0;
	}

	return cuArray;
}

cudaArray* allocateProjectionArray_LayeredAngle(const SDimensions3D& dims)
{
	//fprintf(stderr, "allocateProjectionArray_LayeredAngle: %dx%d, %d angles\n", dims.iProjU, dims.iProjV, dims.iProjAngles);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* cuArray;
	cudaExtent extentA;
	extentA.width = dims.iProjU;
	extentA.height = dims.iProjV;
	extentA.depth = dims.iProjAngles;

	if (!checkCuda(cudaMalloc3DArray(&cuArray, &channelDesc, extentA, cudaArrayLayered), "allocateProjectionArray_LayeredAngle 3D")) {
		ASTRA_ERROR("Failed to allocate layered %dx%dx%d GPU array", dims.iProjU, dims.iProjV, dims.iProjAngles);
		return 0;
	}

	return cuArray;
}

bool transferVolumeToArray(cudaPitchedPtr D_volumeData, cudaArray* array, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

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

	bool ok = checkCuda(cudaMemcpy3DAsync(&p, stream()), "transferVolumeToArray 3D");
	ok &= stream.syncIfSync("transferVolumeToArray 3D sync");
	return ok;
}

bool transferProjectionsToArray(cudaPitchedPtr D_projData, cudaArray* array, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

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

	bool ok = checkCuda(cudaMemcpy3DAsync(&p, stream()), "transferProjectionsToArray 3D");
	ok &= stream.syncIfSync("transferProjectionsToArray 3D sync");
	return ok;
}

bool transferProjectionsToArray_LayeredAngle(cudaPitchedPtr D_projData, cudaArray* array, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	cudaExtent extentA;
	extentA.width = dims.iProjU;
	extentA.height = dims.iProjV;
	extentA.depth = dims.iProjAngles;
	//fprintf(stderr, "transfer %d %d %d pitch %d\n", dims.iProjU, dims.iProjV, dims.iProjAngles, D_projData.pitch);

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

	bool ok = checkCuda(cudaMemcpy3DAsync(&p, stream()), "transferProjectionsToArray_LayeredAngle 3D");
	ok &= stream.syncIfSync("transferProjectionsToArray_LayeredAngle 3D sync");
	return ok;
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

int calcNextPowerOfTwo(int _iValue)
{
	int iOutput = 1;
	while(iOutput < _iValue)
		iOutput *= 2;
	return iOutput;
}

template<bool INV>
__global__ void dev_permute_XYZ_to_ZXY_subX(void* D_outData, void* D_volData,
                                            unsigned int volPitchOut, unsigned int volPitchIn,
                                            unsigned int startX,
                                            const SDimensions3D bufdims, const SDimensions3D dims)
{
	// X <- Y, Y <- Z, Z <- X
	float* volData = (float*)D_volData;
	float* outData = (float*)D_outData;
	int volX = blockIdx.z * g_permuteBlockSlices + threadIdx.y;
	if (startX + volX >= dims.iVolX)
		return;
	if (volX >= bufdims.iVolZ)
		return;

	const int volY = blockIdx.x * g_permuteBlock1 + threadIdx.x;
	if (volY > dims.iVolY)
		return;
	const int startVolZ = blockIdx.y * g_permuteBlock2;
	int endVolZ = startVolZ + g_permuteBlock2;
	if (endVolZ > dims.iVolZ)
		endVolZ = dims.iVolZ;

	for (int volZ = startVolZ; volZ < endVolZ; ++volZ)
	{
		if constexpr (!INV)
			outData[(size_t)(volX*bufdims.iVolY+volZ)*volPitchOut+volY] = volData[(size_t)(volZ*dims.iVolY+volY)*volPitchIn+startX+volX];
		else
			volData[(size_t)(volZ*dims.iVolY+volY)*volPitchIn+startX+volX] += outData[(size_t)(volX*bufdims.iVolY+volZ)*volPitchOut+volY];
	}
}

// If 'invert', do the inverse permutation and ADD the result into the buffer
bool permute_XYZ_to_ZXY_subX(cudaPitchedPtr D_outData, cudaPitchedPtr D_volData,
                             int startX,
                             const SDimensions3D& bufdims, const SDimensions3D& dims,
                             bool invert,
                             std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	assert(dims.iVolY == bufdims.iVolX);
	assert(dims.iVolZ == bufdims.iVolY);

	dim3 dimBlock(g_permuteBlock1, g_permuteBlockSlices);
	dim3 dimGrid = computeGrid3Ceil(bufdims.iVolX, g_permuteBlock1,
	                                bufdims.iVolY, g_permuteBlock2,
	                                bufdims.iVolZ, g_permuteBlockSlices);

	int volPitchOut = D_outData.pitch/sizeof(float);
	int volPitchIn = D_volData.pitch/sizeof(float);

	if (!invert)
		dev_permute_XYZ_to_ZXY_subX<false><<<dimGrid, dimBlock, 0, stream()>>>(D_outData.ptr, D_volData.ptr, volPitchOut, volPitchIn, startX, bufdims, dims);
	else
		dev_permute_XYZ_to_ZXY_subX<true><<<dimGrid, dimBlock, 0, stream()>>>(D_outData.ptr, D_volData.ptr, volPitchOut, volPitchIn, startX, bufdims, dims);

	if (!stream.syncIfSync("permute_XYZ_to_ZXY_subX sync")) {
		return false;
	}

	return true;
}


template<bool INV>
__global__ void dev_permute_XYZ_to_XZY_subY(void* D_outData, void* D_volData,
                                            unsigned int volPitchOut, unsigned int volPitchIn,
                                            unsigned int startY,
                                            const SDimensions3D bufdims, const SDimensions3D dims)
{
	// X <- X, Y <- Z, Z <- Y
	float* volData = (float*)D_volData;
	float* outData = (float*)D_outData;
	int volY = blockIdx.z * g_permuteBlockSlices + threadIdx.y;
	if (startY + volY >= dims.iVolY)
		return;
	if (volY >= bufdims.iVolZ)
		return;

	const int volX = blockIdx.x * g_permuteBlock1 + threadIdx.x;
	if (volX > dims.iVolX)
		return;
	const int startVolZ = blockIdx.y * g_permuteBlock2;
	int endVolZ = startVolZ + g_permuteBlock2;
	if (endVolZ > dims.iVolZ)
		endVolZ = dims.iVolZ;

	for (int volZ = startVolZ; volZ < endVolZ; ++volZ)
	{
		if constexpr (!INV)
			outData[(size_t)(volY*bufdims.iVolY+volZ)*volPitchOut+volX] = volData[(size_t)(volZ*dims.iVolY+startY+volY)*volPitchIn+volX];
		else
			volData[(size_t)(volZ*dims.iVolY+startY+volY)*volPitchIn+volX] += outData[(size_t)(volY*bufdims.iVolY+volZ)*volPitchOut+volX];
	}
}

// If 'invert', do the inverse permutation and ADD the result into the buffer
bool permute_XYZ_to_XZY_subY(cudaPitchedPtr D_outData, cudaPitchedPtr D_volData,
                             int startY,
                             const SDimensions3D& bufdims, const SDimensions3D& dims,
                             bool invert,
                             std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	assert(dims.iVolX == bufdims.iVolX);
	assert(dims.iVolZ == bufdims.iVolY);

	dim3 dimBlock(g_permuteBlock1, g_permuteBlockSlices);
	dim3 dimGrid = computeGrid3Ceil(bufdims.iVolX, g_permuteBlock1,
	                                bufdims.iVolY, g_permuteBlock2,
	                                bufdims.iVolZ, g_permuteBlockSlices);

	int volPitchOut = D_outData.pitch/sizeof(float);
	int volPitchIn = D_volData.pitch/sizeof(float);

	if (!invert)
		dev_permute_XYZ_to_XZY_subY<false><<<dimGrid, dimBlock, 0, stream()>>>(D_outData.ptr, D_volData.ptr, volPitchOut, volPitchIn, startY, bufdims, dims);
	else
		dev_permute_XYZ_to_XZY_subY<true><<<dimGrid, dimBlock, 0, stream()>>>(D_outData.ptr, D_volData.ptr, volPitchOut, volPitchIn, startY, bufdims, dims);

	if (!stream.syncIfSync("permute_XYZ_to_XZY_subY sync")) {
		return false;
	}

	return true;
}


template<bool INV>
__global__ void dev_permute_UTV_to_UVT_subT(void* D_outData, void* D_projData,
                                            unsigned int projPitchOut, unsigned int projPitchIn,
                                            unsigned int startT,
                                            const SDimensions3D bufdims, const SDimensions3D dims)
{
	// X <- X, Y <- Z, Z <- Y
	float* projData = (float*)D_projData;
	float* outData = (float*)D_outData;
	int projT = blockIdx.z * g_permuteBlockSlices + threadIdx.y;
	if (startT + projT >= dims.iProjAngles)
		return;
	if (projT >= bufdims.iProjAngles)
		return;

	const int projU = blockIdx.x * g_permuteBlock1 + threadIdx.x;
	if (projU > dims.iProjU)
		return;
	const int startProjV = blockIdx.y * g_permuteBlock2;
	int endProjV = startProjV + g_permuteBlock2;
	if (endProjV > dims.iProjV)
		endProjV = dims.iProjV;

	for (int projV = startProjV; projV < endProjV; ++projV)
	{
		if constexpr (!INV)
			outData[(size_t)(projT*bufdims.iProjV+projV)*projPitchOut+projU] = projData[(size_t)(projV*dims.iProjAngles+startT+projT)*projPitchIn+projU];
		else
			projData[(size_t)(projV*dims.iProjAngles+startT+projT)*projPitchIn+projU] += outData[(size_t)(projT*bufdims.iProjV+projV)*projPitchOut+projU];
	}
}

// If 'invert', do the inverse permutation and ADD the result into the buffer
// NB: Unlike the volume case, bufdims is NOT permuted. Only bufdims.iProjAngles
// is changed.
bool permute_UTV_to_UVT_subT(cudaPitchedPtr D_outData, cudaPitchedPtr D_projData,
                             int startT,
                             const SDimensions3D& bufdims, const SDimensions3D& dims,
                             bool invert,
                             std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	assert(dims.iProjU == bufdims.iProjU);
	assert(dims.iProjV == bufdims.iProjV);

	dim3 dimBlock(g_permuteBlock1, g_permuteBlockSlices);
	dim3 dimGrid = computeGrid3Ceil(bufdims.iProjU, g_permuteBlock1,
	                                bufdims.iProjV, g_permuteBlock2,
	                                bufdims.iProjAngles, g_permuteBlockSlices);

	int projPitchOut = D_outData.pitch/sizeof(float);
	int projPitchIn = D_projData.pitch/sizeof(float);

	if (!invert)
		dev_permute_UTV_to_UVT_subT<false><<<dimGrid, dimBlock, 0, stream()>>>(D_outData.ptr, D_projData.ptr, projPitchOut, projPitchIn, startT, bufdims, dims);
	else
		dev_permute_UTV_to_UVT_subT<true><<<dimGrid, dimBlock, 0, stream()>>>(D_outData.ptr, D_projData.ptr, projPitchOut, projPitchIn, startT, bufdims, dims);

	if (!stream.syncIfSync("permute_UTV_to_UVT_subT sync")) {
		return false;
	}

	return true;
}

}
