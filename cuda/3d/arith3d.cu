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

#include "util3d.h"
#include "arith3d.h"
#include <cassert>

namespace astraCUDA3d {

struct opAddScaled {
	__device__ void operator()(float& out, const float in, const float inp) {
		out += in * inp;
	}
};
struct opScaleAndAdd {
	__device__ void operator()(float& out, const float in, const float inp) {
		out = in + out * inp;
	}
};
struct opAddMulScaled {
	__device__ void operator()(float& out, const float in1, const float in2, const float inp) {
		out += in1 * in2 * inp;
	}
};
struct opAddMul {
	__device__ void operator()(float& out, const float in1, const float in2) {
		out += in1 * in2;
	}
};
struct opAdd {
	__device__ void operator()(float& out, const float in) {
		out += in;
	}
};
struct opMul {
	__device__ void operator()(float& out, const float in) {
		out *= in;
	}
};
struct opMul2 {
	__device__ void operator()(float& out, const float in1, const float in2) {
		out *= in1 * in2;
	}
};
struct opDividedBy {
	__device__ void operator()(float& out, const float in) {
		if (out > 0.000001f) // out is assumed to be positive
			out = in / out;
		else
			out = 0.0f;
	}
};
struct opInvert {
	__device__ void operator()(float& out) {
		if (out > 0.000001f) // out is assumed to be positive
			out = 1 / out;
		else
			out = 0.0f;
	}
};
struct opSet {
	__device__ void operator()(float& out, const float inp) {
		out = inp;
	}
};
struct opClampMin {
	__device__ void operator()(float& out, const float inp) {
		if (out < inp)
			out = inp;
	}
};
struct opClampMax {
	__device__ void operator()(float& out, const float inp) {
		if (out > inp)
			out = inp;
	}
};




template<class op, unsigned int repeat>
__global__ void devtoD(float* pfOut, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off]);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int repeat>
__global__ void devFtoD(float* pfOut, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], fParam);
		off += pitch;
		y++;
	}
}


template<class op, unsigned int repeat>
__global__ void devDtoD(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn[off]);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int repeat>
__global__ void devDFtoD(float* pfOut, const float* pfIn, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn[off], fParam);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int repeat>
__global__ void devDDtoD(float* pfOut, const float* pfIn1, const float* pfIn2, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn1[off], pfIn2[off]);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int repeat>
__global__ void devDDFtoD(float* pfOut, const float* pfIn1, const float* pfIn2, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn1[off], pfIn2[off], fParam);
		off += pitch;
		y++;
	}
}









template<typename op>
void processVol(CUdeviceptr* out, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+511)/512);

	float *pfOut = (float*)out;

	devtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol(CUdeviceptr* out, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	float *pfOut = (float*)out;

	devFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol(CUdeviceptr* out, const CUdeviceptr* in, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	float *pfOut = (float*)out;
	const float *pfIn = (const float*)in;

	devDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol(CUdeviceptr* out, const CUdeviceptr* in, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	float *pfOut = (float*)out;
	const float *pfIn = (const float*)in;

	devDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol(CUdeviceptr* out, const CUdeviceptr* in1, const CUdeviceptr* in2, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	float *pfOut = (float*)out;
	const float *pfIn1 = (const float*)in1;
	const float *pfIn2 = (const float*)in2;

	devDDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol(CUdeviceptr* out, const CUdeviceptr* in1, const CUdeviceptr* in2, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	float *pfOut = (float*)out;
	const float *pfIn1 = (const float*)in1;
	const float *pfIn2 = (const float*)in2;

	devDDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

















template<typename op>
void processVol3D(cudaPitchedPtr& out, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devtoD<op, 32><<<gridSize, blockSize>>>(pfOut, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn = (float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn = (float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn1 = (float*)in1.ptr;
	float *pfIn2 = (float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn1 = (float*)in1.ptr;
	float *pfIn2 = (float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	cudaTextForceKernelsCompletion();
}













template<typename op>
void processSino3D(cudaPitchedPtr& out, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devtoD<op, 32><<<gridSize, blockSize>>>(pfOut, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processSino3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn = (float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn = (float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn1 = (float*)in1.ptr;
	float *pfIn2 = (float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	float *pfIn1 = (float*)in1.ptr;
	float *pfIn2 = (float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	cudaTextForceKernelsCompletion();
}


















#define INST_DFtoD(name) \
  template void processVol<name>(CUdeviceptr* out, const CUdeviceptr* in, float fParam, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims);

#define INST_DtoD(name) \
  template void processVol<name>(CUdeviceptr* out, const CUdeviceptr* in, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims);

#define INST_DDtoD(name) \
  template void processVol<name>(CUdeviceptr* out, const CUdeviceptr* in1, const CUdeviceptr* in2, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims);

#define INST_DDFtoD(name) \
  template void processVol<name>(CUdeviceptr* out, const CUdeviceptr* in1, const CUdeviceptr* in2, float fParam, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims);


#define INST_toD(name) \
  template void processVol<name>(CUdeviceptr* out, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const SDimensions3D& dims);

#define INST_FtoD(name) \
  template void processVol<name>(CUdeviceptr* out, float fParam, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims);



INST_DFtoD(opAddScaled)
INST_DFtoD(opScaleAndAdd)
INST_DDFtoD(opAddMulScaled)
INST_DDtoD(opAddMul)
INST_DDtoD(opMul2)
INST_DtoD(opMul)
INST_DtoD(opAdd)
INST_DtoD(opDividedBy)
INST_toD(opInvert)
INST_FtoD(opMul)
INST_FtoD(opSet)
INST_FtoD(opClampMin)
INST_FtoD(opClampMax)


}
