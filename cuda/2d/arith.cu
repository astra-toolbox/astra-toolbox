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



template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devtoD(float* pfOut, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off]);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devFtoD(float* pfOut, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], fParam);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devFFtoDD(float* pfOut1, float* pfOut2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut1[off], pfOut2[off], fParam1, fParam2);
		off += pitch;
		y++;
	}
}



template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devDtoD(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn[off]);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devDFtoD(float* pfOut, const float* pfIn, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn[off], fParam);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devDDtoD(float* pfOut, const float* pfIn1, const float* pfIn2, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn1[off], pfIn2[off]);
		off += pitch;
		y++;
	}
}

template<class op, unsigned int padX, unsigned int padY, unsigned int repeat>
__global__ void devDDFtoD(float* pfOut, const float* pfIn1, const float* pfIn2, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	unsigned int y = (threadIdx.y + 16*blockIdx.y)*repeat;
	unsigned int off = (y+padY)*pitch+x+padX;
	for (unsigned int i = 0; i < repeat; ++i) {
		if (y >= height)
			break;
		op()(pfOut[off], pfIn1[off], pfIn2[off], fParam);
		off += pitch;
		y++;
	}
}
















template<typename op, VolType t>
void processVolCopy(float* out, unsigned int width, unsigned int height)
{
	float* D_out;

	unsigned int pitch;
	allocateVolume(D_out, width+2, height+2, pitch);
	copyVolumeToDevice(out, width, width, height, D_out, pitch);

	processVol<op, t>(D_out, pitch, width, height);

	copyVolumeFromDevice(out, width, width, height, D_out, pitch);

	cudaFree(D_out);
}

template<typename op, VolType t>
void processVolCopy(float* out, float param, unsigned int width, unsigned int height)
{
	float* D_out;

	unsigned int pitch;
	allocateVolume(D_out, width+2, height+2, pitch);
	copyVolumeToDevice(out, width, width, height, D_out, pitch);

	processVol<op, t>(D_out, param, pitch, width, height);

	copyVolumeFromDevice(out, width, width, height, D_out, pitch);

	cudaFree(D_out);
}

template<typename op, VolType t>
void processVolCopy(float* out1, float* out2, float param1, float param2, unsigned int width, unsigned int height)
{
	float* D_out1;
	float* D_out2;

	unsigned int pitch;
	allocateVolume(D_out1, width+2, height+2, pitch);
	copyVolumeToDevice(out1, width, width, height, D_out1, pitch);
	allocateVolume(D_out2, width+2, height+2, pitch);
	copyVolumeToDevice(out2, width, width, height, D_out2, pitch);

	processVol<op, t>(D_out1, D_out2, param1, param2, pitch, width, height);

	copyVolumeFromDevice(out1, width, width, height, D_out1, pitch);
	copyVolumeFromDevice(out2, width, width, height, D_out2, pitch);

	cudaFree(D_out1);
	cudaFree(D_out2);
}


template<typename op, VolType t>
void processVolCopy(float* out, const float* in, unsigned int width, unsigned int height)
{
	float* D_out;
	float* D_in;

	unsigned int pitch;
	allocateVolume(D_out, width+2, height+2, pitch);
	copyVolumeToDevice(out, width, width, height, D_out, pitch);
	allocateVolume(D_in, width+2, height+2, pitch);
	copyVolumeToDevice(in, width, width, height, D_in, pitch);

	processVol<op, t>(D_out, D_in, pitch, width, height);

	copyVolumeFromDevice(out, width, width, height, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in);
}

template<typename op, VolType t>
void processVolCopy(float* out, const float* in, float param, unsigned int width, unsigned int height)
{
	float* D_out;
	float* D_in;

	unsigned int pitch;
	allocateVolume(D_out, width+2, height+2, pitch);
	copyVolumeToDevice(out, width, width, height, D_out, pitch);
	allocateVolume(D_in, width+2, height+2, pitch);
	copyVolumeToDevice(in, width, width, height, D_in, pitch);

	processVol<op, t>(D_out, D_in, param, pitch, width, height);

	copyVolumeFromDevice(out, width, width, height, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in);
}

template<typename op, VolType t>
void processVolCopy(float* out, const float* in1, const float* in2, unsigned int width, unsigned int height)
{
	float* D_out;
	float* D_in1;
	float* D_in2;

	unsigned int pitch;
	allocateVolume(D_out, width+2, height+2, pitch);
	copyVolumeToDevice(out, width, width, height, D_out, pitch);
	allocateVolume(D_in1, width+2, height+2, pitch);
	copyVolumeToDevice(in1, width, width, height, D_in1, pitch);
	allocateVolume(D_in2, width+2, height+2, pitch);
	copyVolumeToDevice(in2, width, width, height, D_in2, pitch);

	processVol<op, t>(D_out, D_in1, D_in2, pitch, width, height);

	copyVolumeFromDevice(out, width, width, height, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in1);
	cudaFree(D_in2);
}

template<typename op, VolType t>
void processVolCopy(float* out, const float* in1, const float* in2, float param, unsigned int width, unsigned int height)
{
	float* D_out;
	float* D_in1;
	float* D_in2;

	unsigned int pitch;
	allocateVolume(D_out, width+2, height+2, pitch);
	copyVolumeToDevice(out, width, width, height, D_out, pitch);
	allocateVolume(D_in1, width+2, height+2, pitch);
	copyVolumeToDevice(in1, width, width, height, D_in1, pitch);
	allocateVolume(D_in2, width+2, height+2, pitch);
	copyVolumeToDevice(in2, width, width, height, D_in2, pitch);

	processVol<op, t>(D_out, D_in1, D_in2, param, pitch, width, height);

	copyVolumeFromDevice(out, width, width, height, D_out, pitch);

	cudaFree(D_out);
	cudaFree(D_in1);
	cudaFree(D_in2);
}









template<typename op, VolType t>
void processVol(float* pfOut, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+511)/512);

	devtoD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op, VolType t>
void processVol(float* pfOut, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devFtoD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op, VolType t>
void processVol(float* pfOut1, float* pfOut2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devFFtoDD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut1, pfOut2, fParam1, fParam2, pitch, width, height);

	cudaTextForceKernelsCompletion();
}


template<typename op, VolType t>
void processVol(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDtoD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut, pfIn, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op, VolType t>
void processVol(float* pfOut, const float* pfIn, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDFtoD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op, VolType t>
void processVol(float* pfOut, const float* pfIn1, const float* pfIn2, float fParam, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDDFtoD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, pitch, width, height);

	cudaTextForceKernelsCompletion();
}

template<typename op, VolType t>
void processVol(float* pfOut, const float* pfIn1, const float* pfIn2, unsigned int pitch, unsigned int width, unsigned int height)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	devDDtoD<op, 1, t, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, pitch, width, height);

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
		devtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
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
		devFtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processVol3D(cudaPitchedPtr& out1, cudaPitchedPtr& out2, float fParam1, float fParam2, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut1 = (float*)out1.ptr;
	float *pfOut2 = (float*)out2.ptr;
	unsigned int step = out1.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devFFtoDD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut1, pfOut2, fParam1, fParam2, out1.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut1 += step;
		pfOut2 += step;
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
		devDtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
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
		devDFtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
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
		devDDFtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
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
		devDDtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
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
		devtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
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
		devFtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
	}

	cudaTextForceKernelsCompletion();
}

template<typename op>
void processSino3D(cudaPitchedPtr& out1, cudaPitchedPtr& out2, float fParam1, float fParam2, const SDimensions3D& dims)
{
	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut1 = (float*)out1.ptr;
	float *pfOut2 = (float*)out2.ptr;
	unsigned int step = out1.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devFFtoDD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut1, pfOut2, fParam1, fParam2, out1.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut1 += step;
		pfOut2 += step;
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
		devDtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
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
		devDFtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
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
		devDDFtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
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
		devDDtoD<op, 0, 0, 32><<<gridSize, blockSize>>>(pfOut, pfIn1, pfIn2, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	cudaTextForceKernelsCompletion();
}


















#define INST_DFtoD(name) \
  template void processVolCopy<name, VOL>(float* out, const float* in, float param, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out, const float* in, float param, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out, const float* in, float param, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out, const float* in, float param, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims);

#define INST_DtoD(name) \
  template void processVolCopy<name, VOL>(float* out, const float* in, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out, const float* in, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out, const float* in, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out, const float* in, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims);

#define INST_DDtoD(name) \
  template void processVolCopy<name, VOL>(float* out, const float* in1, const float* in2, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out, const float* in1, const float* in2, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out, const float* in1, const float* in2, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out, const float* in1, const float* in2, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims);

#define INST_DDFtoD(name) \
  template void processVolCopy<name, VOL>(float* out, const float* in1, const float* in2, float fParam, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out, const float* in1, const float* in2, float fParam, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims);


#define INST_toD(name) \
  template void processVolCopy<name, VOL>(float* out, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, const SDimensions3D& dims);

#define INST_FtoD(name) \
  template void processVolCopy<name, VOL>(float* out, float param, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out, float param, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out, float param, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out, float param, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out, float param, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out, float param, const SDimensions3D& dims);

#define INST_FFtoDD(name) \
  template void processVolCopy<name, VOL>(float* out1, float* out2, float fParam1, float fParam2, unsigned int width, unsigned int height); \
  template void processVolCopy<name, SINO>(float* out1, float* out2, float fParam1, float fParam2, unsigned int width, unsigned int height); \
  template void processVol<name, VOL>(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol<name, SINO>(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height); \
  template void processVol3D<name>(cudaPitchedPtr& out1, cudaPitchedPtr& out2, float fParam1, float fParam2, const SDimensions3D& dims); \
  template void processSino3D<name>(cudaPitchedPtr& out1, cudaPitchedPtr& out2, float fParam1, float fParam2, const SDimensions3D& dims);



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
