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
#include "astra/cuda/3d/arith3d.h"

#include "astra/cuda/2d/util.h"

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
bool processVol3D(cudaPitchedPtr& out, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processVol3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devFtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn = (const float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn = (const float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDFtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn1 = (const float*)in1.ptr;
	const float *pfIn2 = (const float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDDFtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn1, pfIn2, fParam, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn1 = (const float*)in1.ptr;
	const float *pfIn2 = (const float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iVolY;

	for (unsigned int i = 0; i < dims.iVolZ; ++i) {
		devDDtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn1, pfIn2, out.pitch/sizeof(float), dims.iVolX, dims.iVolY);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}













template<typename op>
bool processSino3D(cudaPitchedPtr& out, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processSino3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devFtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn = (const float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn = (const float*)in.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDFtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn1 = (const float*)in1.ptr;
	const float *pfIn2 = (const float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDDFtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn1, pfIn2, fParam, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}

template<typename op>
bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims, std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	dim3 blockSize(16,16);
	dim3 gridSize((dims.iProjU+15)/16, (dims.iProjAngles+511)/512);
	float *pfOut = (float*)out.ptr;
	const float *pfIn1 = (const float*)in1.ptr;
	const float *pfIn2 = (const float*)in2.ptr;
	unsigned int step = out.pitch/sizeof(float) * dims.iProjAngles;

	for (unsigned int i = 0; i < dims.iProjV; ++i) {
		devDDtoD<op, 32><<<gridSize, blockSize, 0, stream()>>>(pfOut, pfIn1, pfIn2, out.pitch/sizeof(float), dims.iProjU, dims.iProjAngles);
		pfOut += step;
		pfIn1 += step;
		pfIn2 += step;
	}

	return stream.syncIfSync(__FUNCTION__);
}


















#define INST_DFtoD(name) \
  template bool processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream); \
  template bool processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream);

#define INST_DtoD(name) \
  template bool processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims, std::optional<cudaStream_t> _stream); \
  template bool processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims, std::optional<cudaStream_t> _stream);

#define INST_DDtoD(name) \
  template bool processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims, std::optional<cudaStream_t> _stream); \
  template bool processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims, std::optional<cudaStream_t> _stream);

#define INST_DDFtoD(name) \
  template bool processVol3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream); \
  template bool processSino3D<name>(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream);


#define INST_toD(name) \
  template bool processVol3D<name>(cudaPitchedPtr& out, const SDimensions3D& dims, std::optional<cudaStream_t> _stream); \
  template bool processSino3D<name>(cudaPitchedPtr& out, const SDimensions3D& dims, std::optional<cudaStream_t> _stream);

#define INST_FtoD(name) \
  template bool processVol3D<name>(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream); \
  template bool processSino3D<name>(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream);



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
