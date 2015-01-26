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

#include "util.h"
#include "arith.h"
#include <cassert>

namespace astraCUDA {


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
struct opAdd2 {
	__device__ void operator()(float& out, const float in1, const float in2) {
		out += in1 + in2;
	}
};
struct opMul {
	__device__ void operator()(float& out, const float in) {
		out *= in;
	}
};
struct opDiv {
	__device__ void operator()(float& out, const float in) {
		if (in > 0.000001f) // out is assumed to be positive
			out /= in;
		else
			out = 0.0f;
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
struct opClampMinMask {
	__device__ void operator()(float& out, const float in) {
		if (out < in)
			out = in;
	}
};
struct opClampMaxMask {
	__device__ void operator()(float& out, const float in) {
		if (out > in)
			out = in;
	}
};
struct opSetMaskedValues {
	__device__ void operator()(float& out, const float in, const float inp) {
		if (!in)
			out = inp;
	}
};
struct opSegmentAndMask {
	__device__ void operator()(float& out1, float& out2, const float inp1, const float inp2) {
		if (out1 >= inp1) {
			out1 = inp2;
			out2 = 0.0f;
		}

	}

};
struct opMulMask {
	__device__ void operator()(float& out, const float mask, const float in) {
		if (mask > 0.0f) {
			out *= in;
		}
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
__global__ void devFFtoDD(float* pfOut1, float* pfOut2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = y*pitch+x;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut1[off], pfOut2[off], fParam1, fParam2);
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
void processVolCopy(float* out, const SDimensions& dims)
{
	float* D_out;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_out, pitch);

	processVol<op>(D_out, pitch, dims);

	copyVolumeFromDevice(out, width, dims, D_out, pitch);

	cudaFree(D_out);
}

template<typename op>
void processVolCopy(float* out, float param, const SDimensions& dims)
{
	float* D_out;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_out, pitch);

	processVol<op>(D_out, param, pitch, dims);

	copyVolumeFromDevice(out, width, dims, D_out, pitch);

	cudaFree(D_out);
}

template<typename op>
void processVolCopy(float* out1, float* out2, float param1, float param2, const SDimensions& dims)
{
	float* D_out1;
	float* D_out2;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out1, pitch, dims);
	copyVolumeToDevice(out1, width, dims, D_out1, pitch);
	allocateVolumeData(D_out2, pitch, dims);
	copyVolumeToDevice(out2, width, dims, D_out2, pitch);

	processVol<op>(D_out1, D_out2, param1, param2, pitch, dims);

	copyVolumeFromDevice(out1, width, dims, D_out1, pitch);
	copyVolumeFromDevice(out2, width, dims, D_out2, pitch);

	cudaFree(D_out1);
	cudaFree(D_out2);
}


template<typename op>
void processVolCopy(float* out, const float* in, const SDimensions& dims)
{
	float* D_out;
	float* D_in;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_out, pitch);
	allocateVolumeData(D_in, pitch, dims);
	copyVolumeToDevice(in, width, dims, D_in, pitch);

	processVol<op>(D_out, D_in, pitch, dims);

	copyVolumeFromDevice(out, width, dims, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in);
}

template<typename op>
void processVolCopy(float* out, const float* in, float param, const SDimensions& dims)
{
	float* D_out;
	float* D_in;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_out, pitch);
	allocateVolumeData(D_in, pitch, dims);
	copyVolumeToDevice(in, width, dims, D_in, pitch);

	processVol<op>(D_out, D_in, param, pitch, dims);

	copyVolumeFromDevice(out, width, dims, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in);
}

template<typename op>
void processVolCopy(float* out, const float* in1, const float* in2, const SDimensions& dims)
{
	float* D_out;
	float* D_in1;
	float* D_in2;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_out, pitch);
	allocateVolumeData(D_in1, pitch, dims);
	copyVolumeToDevice(in1, width, dims, D_in1, pitch);
	allocateVolumeData(D_in2, pitch, dims);
	copyVolumeToDevice(in2, width, dims, D_in2, pitch);

	processVol<op>(D_out, D_in1, D_in2, pitch, dims);

	copyVolumeFromDevice(out, width, dims, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in1);
	cudaFree(D_in2);
}

template<typename op>
void processVolCopy(float* out, const float* in1, const float* in2, float param, const SDimensions& dims)
{
	float* D_out;
	float* D_in1;
	float* D_in2;
	size_t width = dims.iVolWidth;

	unsigned int pitch;
	allocateVolumeData(D_out, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_out, pitch);
	allocateVolumeData(D_in1, pitch, dims);
	copyVolumeToDevice(in1, width, dims, D_in1, pitch);
	allocateVolumeData(D_in2, pitch, dims);
	copyVolumeToDevice(in2, width, dims, D_in2, pitch);

	processVol<op>(D_out, D_in1, D_in2, param, pitch, dims);

	copyVolumeFromDevice(out, width, dims, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in1);
	cudaFree(D_in2);
}








template<typename op>
void processData(float* pfOut, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+511)/512);

	devtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processData(float* pfOut, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processData(float* pfOut1, float* pfOut2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devFFtoDD<op, 32><<<gridSize, blockSize>>>(pfOut1, pfOut2, fParam1, fParam2, pitch, width, height);

	cudaTextForceKernelsCompletion();
}


template<typename op>
void processData(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processData(float* pfOut, const float* pfIn, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processData(float* pfOut, const float* pfIn1, const float* pfIn2, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDDFtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processData(float* pfOut, const float* pfIn1, const float* pfIn2, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDDtoD<op, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, pitch, width, height);

	cudaTextForceKernelsCompletion();
}








template<typename op>
void processVol(float* out, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, pitch, dims.iVolWidth, dims.iVolHeight);
}

template<typename op>
void processVol(float* out, float param, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, param, pitch, dims.iVolWidth, dims.iVolHeight);
}

template<typename op>
void processVol(float* out1, float* out2, float param1, float param2, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out1, out2, param1, param2, pitch, dims.iVolWidth, dims.iVolHeight);
}


template<typename op>
void processVol(float* out, const float* in, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in, pitch, dims.iVolWidth, dims.iVolHeight);
}

template<typename op>
void processVol(float* out, const float* in, float param, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in, param, pitch, dims.iVolWidth, dims.iVolHeight);
}

template<typename op>
void processVol(float* out, const float* in1, const float* in2, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in1, in2, pitch, dims.iVolWidth, dims.iVolHeight);
}

template<typename op>
void processVol(float* out, const float* in1, const float* in2, float param, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in2, in2, param, pitch, dims.iVolWidth, dims.iVolHeight);
}




template<typename op>
void processSino(float* out, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, pitch, dims.iProjDets, dims.iProjAngles);
}

template<typename op>
void processSino(float* out, float param, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, param, pitch, dims.iProjDets, dims.iProjAngles);
}

template<typename op>
void processSino(float* out1, float* out2, float param1, float param2, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out1, out2, param1, param2, pitch, dims.iProjDets, dims.iProjAngles);
}


template<typename op>
void processSino(float* out, const float* in, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in, pitch, dims.iProjDets, dims.iProjAngles);
}

template<typename op>
void processSino(float* out, const float* in, float param, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in, param, pitch, dims.iProjDets, dims.iProjAngles);
}

template<typename op>
void processSino(float* out, const float* in1, const float* in2, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in1, in2, pitch, dims.iProjDets, dims.iProjAngles);
}

template<typename op>
void processSino(float* out, const float* in1, const float* in2, float param, unsigned int pitch, const SDimensions& dims)
{
	processData<op>(out, in2, in2, param, pitch, dims.iProjDets, dims.iProjAngles);
}























#define INST_DFtoD(name) \
  template void processVolCopy<name>(float* out, const float* in, float param, const SDimensions& dims); \
  template void processVol<name>(float* out, const float* in, float param, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out, const float* in, float param, unsigned int pitch, const SDimensions& dims);

#define INST_DtoD(name) \
  template void processVolCopy<name>(float* out, const float* in, const SDimensions& dims); \
  template void processVol<name>(float* out, const float* in, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out, const float* in, unsigned int pitch, const SDimensions& dims);

#define INST_DDtoD(name) \
  template void processVolCopy<name>(float* out, const float* in1, const float* in2, const SDimensions& dims); \
  template void processVol<name>(float* out, const float* in1, const float* in2, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out, const float* in1, const float* in2, unsigned int pitch, const SDimensions& dims);

#define INST_DDFtoD(name) \
  template void processVolCopy<name>(float* out, const float* in1, const float* in2, float fParam, const SDimensions& dims); \
  template void processVol<name>(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, const SDimensions& dims);


#define INST_toD(name) \
  template void processVolCopy<name>(float* out, const SDimensions& dims); \
  template void processVol<name>(float* out, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out, unsigned int pitch, const SDimensions& dims);

#define INST_FtoD(name) \
  template void processVolCopy<name>(float* out, float param, const SDimensions& dims); \
  template void processVol<name>(float* out, float param, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out, float param, unsigned int pitch, const SDimensions& dims);

#define INST_FFtoDD(name) \
  template void processVolCopy<name>(float* out1, float* out2, float fParam1, float fParam2, const SDimensions& dims); \
  template void processVol<name>(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, const SDimensions& dims); \
  template void processSino<name>(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, const SDimensions& dims);



INST_DFtoD(opAddScaled)
INST_DFtoD(opScaleAndAdd)
INST_DDFtoD(opAddMulScaled)
INST_DDtoD(opAddMul)
INST_DDtoD(opMul2)
INST_DDtoD(opAdd2)
INST_DtoD(opMul)
INST_DDtoD(opMulMask)
INST_DtoD(opAdd)
INST_DtoD(opDividedBy)
INST_toD(opInvert)
INST_FtoD(opSet)
INST_FtoD(opMul)
INST_DtoD(opDiv)
INST_DFtoD(opMulMask)
INST_FtoD(opAdd)
INST_FtoD(opClampMin)
INST_FtoD(opClampMax)
INST_DtoD(opClampMinMask)
INST_DtoD(opClampMaxMask)

// PDART-specific:
INST_DFtoD(opSetMaskedValues)
INST_FFtoDD(opSegmentAndMask)

}
