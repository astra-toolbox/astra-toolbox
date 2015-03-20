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
#include "util.h"

#include "../../include/astra/Logging.h"

namespace astraCUDA {

bool copyVolumeToDevice(const float* in_data, unsigned int in_pitch,
		const SDimensions& dims,
		float* outD_data, unsigned int out_pitch)
{
	size_t width = dims.iVolWidth;
	size_t height = dims.iVolHeight;
	// TODO: memory order
	cudaError_t err;
	err = cudaMemcpy2D(outD_data, sizeof(float)*out_pitch, in_data, sizeof(float)*in_pitch, sizeof(float)*width, height, cudaMemcpyHostToDevice);
	ASTRA_CUDA_ASSERT(err);
	assert(err == cudaSuccess);
	return true;
}

bool copyVolumeFromDevice(float* out_data, unsigned int out_pitch,
		const SDimensions& dims,
		float* inD_data, unsigned int in_pitch)
{
	size_t width = dims.iVolWidth;
	size_t height = dims.iVolHeight;
	// TODO: memory order
	cudaError_t err = cudaMemcpy2D(out_data, sizeof(float)*out_pitch, inD_data, sizeof(float)*in_pitch, sizeof(float)*width, height, cudaMemcpyDeviceToHost);
	ASTRA_CUDA_ASSERT(err);
	return true;
}


bool copySinogramFromDevice(float* out_data, unsigned int out_pitch,
		const SDimensions& dims,
		float* inD_data, unsigned int in_pitch)
{
	size_t width = dims.iProjDets;
	size_t height = dims.iProjAngles;
	// TODO: memory order
	cudaError_t err = cudaMemcpy2D(out_data, sizeof(float)*out_pitch, inD_data, sizeof(float)*in_pitch, sizeof(float)*width, height, cudaMemcpyDeviceToHost);
	ASTRA_CUDA_ASSERT(err);
	return true;
}

bool copySinogramToDevice(const float* in_data, unsigned int in_pitch,
		const SDimensions& dims,
		float* outD_data, unsigned int out_pitch)
{
	size_t width = dims.iProjDets;
	size_t height = dims.iProjAngles;
	// TODO: memory order
	cudaError_t err;
	err = cudaMemcpy2D(outD_data, sizeof(float)*out_pitch, in_data, sizeof(float)*in_pitch, sizeof(float)*width, height, cudaMemcpyHostToDevice);
	ASTRA_CUDA_ASSERT(err);
	return true;
}


bool allocateVolume(float*& ptr, unsigned int width, unsigned int height, unsigned int& pitch)
{
	size_t p;
	cudaError_t ret = cudaMallocPitch((void**)&ptr, &p, sizeof(float)*width, height);
	if (ret != cudaSuccess) {
		reportCudaError(ret);
		ASTRA_ERROR("Failed to allocate %dx%d GPU buffer", width, height);
		return false;
	}

	assert(p % sizeof(float) == 0);

	pitch = p / sizeof(float);

	return true;
}

void zeroVolume(float* data, unsigned int pitch, unsigned int width, unsigned int height)
{
	cudaError_t err;
	err = cudaMemset2D(data, sizeof(float)*pitch, 0, sizeof(float)*width, height);
	ASTRA_CUDA_ASSERT(err);
}

bool allocateVolumeData(float*& D_ptr, unsigned int& pitch, const SDimensions& dims)
{
	return allocateVolume(D_ptr, dims.iVolWidth, dims.iVolHeight, pitch);
}

bool allocateProjectionData(float*& D_ptr, unsigned int& pitch, const SDimensions& dims)
{
	return allocateVolume(D_ptr, dims.iProjDets, dims.iProjAngles, pitch);
}

void zeroVolumeData(float* D_ptr, unsigned int pitch, const SDimensions& dims)
{
	zeroVolume(D_ptr, pitch, dims.iVolWidth, dims.iVolHeight);
}

void zeroProjectionData(float* D_ptr, unsigned int pitch, const SDimensions& dims)
{
	zeroVolume(D_ptr, pitch, dims.iProjDets, dims.iProjAngles);
}

void duplicateVolumeData(float* D_dst, float* D_src, unsigned int pitch, const SDimensions& dims)
{
	cudaMemcpy2D(D_dst, sizeof(float)*pitch, D_src, sizeof(float)*pitch, sizeof(float)*dims.iVolWidth, dims.iVolHeight, cudaMemcpyDeviceToDevice);
}

void duplicateProjectionData(float* D_dst, float* D_src, unsigned int pitch, const SDimensions& dims)
{
	cudaMemcpy2D(D_dst, sizeof(float)*pitch, D_src, sizeof(float)*pitch, sizeof(float)*dims.iProjDets, dims.iProjAngles, cudaMemcpyDeviceToDevice);
}

template <unsigned int blockSize>
__global__ void reduce1D(float *g_idata, float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;

	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*gridDim.x;
	sdata[tid] = 0;
	while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		volatile float* smem = sdata;
		if (blockSize >= 64) smem[tid] += smem[tid + 32];
		if (blockSize >= 32) smem[tid] += smem[tid + 16];
		if (blockSize >= 16) smem[tid] += smem[tid + 8];
		if (blockSize >= 8) smem[tid] += smem[tid + 4];
		if (blockSize >= 4) smem[tid] += smem[tid + 2];
		if (blockSize >= 2) smem[tid] += smem[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2D(float *g_idata, float *g_odata,
                         unsigned int pitch,
                         unsigned int nx, unsigned int ny)
{
	extern __shared__ float sdata[];
	const unsigned int tidx = threadIdx.x;
	const unsigned int tidy = threadIdx.y;
	const unsigned int tid = tidy * 16 + tidx;

	unsigned int x = blockIdx.x*16 + tidx;
	unsigned int y = blockIdx.y*16 + tidy;

	sdata[tid] = 0;

	if (x < nx) {

		while (y < ny) {
			sdata[tid] += (g_idata[pitch*y+x] * g_idata[pitch*y+x]);
			y += 16 * gridDim.y;
		}

	}

	__syncthreads();

	if (tid < 128)
		sdata[tid] += sdata[tid + 128];
	__syncthreads();

	if (tid < 64)
		sdata[tid] += sdata[tid + 64];
	__syncthreads();

	if (tid < 32) { // 32 is warp size
		volatile float* smem = sdata;
		smem[tid] += smem[tid + 32];
		smem[tid] += smem[tid + 16];
		smem[tid] += smem[tid + 8];
		smem[tid] += smem[tid + 4];
		smem[tid] += smem[tid + 2];
		smem[tid] += smem[tid + 1];
	} 

	if (tid == 0)
		g_odata[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
}

float dotProduct2D(float* D_data, unsigned int pitch,
                   unsigned int width, unsigned int height)
{
	unsigned int bx = (width + 15) / 16;
	unsigned int by = (height + 127) / 128;
	unsigned int shared_mem2 = sizeof(float) * 16 * 16;

	dim3 dimBlock2(16, 16);
	dim3 dimGrid2(bx, by);

	float* D_buf;
	cudaMalloc(&D_buf, sizeof(float) * (bx * by + 1) );
	float* D_res = D_buf + (bx*by);

	// Step 1: reduce 2D from image to a single vector, taking sum of squares

	reduce2D<<< dimGrid2, dimBlock2, shared_mem2>>>(D_data, D_buf, pitch, width, height);
	cudaTextForceKernelsCompletion();

	// Step 2: reduce 1D: add up elements in vector
	if (bx * by > 512)
		reduce1D<512><<< 1, 512, sizeof(float)*512>>>(D_buf, D_res, bx*by);
	else if (bx * by > 128)
		reduce1D<128><<< 1, 128, sizeof(float)*128>>>(D_buf, D_res, bx*by);
	else if (bx * by > 32)
		reduce1D<32><<< 1, 32, sizeof(float)*32*2>>>(D_buf, D_res, bx*by);
	else if (bx * by > 8)
		reduce1D<8><<< 1, 8, sizeof(float)*8*2>>>(D_buf, D_res, bx*by);
	else
		reduce1D<1><<< 1, 1, sizeof(float)*1*2>>>(D_buf, D_res, bx*by);

	float x;
	cudaMemcpy(&x, D_res, 4, cudaMemcpyDeviceToHost);

	cudaTextForceKernelsCompletion();

	cudaFree(D_buf);

	return x;
}


bool cudaTextForceKernelsCompletion()
{
	cudaError_t returnedCudaError = cudaThreadSynchronize();

	if(returnedCudaError != cudaSuccess) {
		ASTRA_ERROR("Failed to force completion of cuda kernels: %d: %s.", returnedCudaError, cudaGetErrorString(returnedCudaError));
		return false;
	}

	return true;
}

void reportCudaError(cudaError_t err)
{
	if(err != cudaSuccess)
		ASTRA_ERROR("CUDA error %d: %s.", err, cudaGetErrorString(err));
}



}
