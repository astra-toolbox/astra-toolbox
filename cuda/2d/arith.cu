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

#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/arith.h"

#include "astra/cuda/stream_internal.h"
#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Data2D.h"

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
bool processData(astra::CData2D *out, const Stream &stream)
{
	CDataGPU *outs = dynamic_cast<CDataGPU*>(out->getStorage());
	assert(outs);
	assert(!outs->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut = (float*)outs->getPtr().ptr;
	unsigned int outPitch = outs->getPtr().pitch / sizeof(float);

	devtoD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}

template<typename op>
bool processData(astra::CData2D *out, float fParam, const Stream &stream)
{
	CDataGPU *outs = dynamic_cast<CDataGPU*>(out->getStorage());
	assert(outs);
	assert(!outs->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut = (float*)outs->getPtr().ptr;
	unsigned int outPitch = outs->getPtr().pitch / sizeof(float);

	devFtoD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut, fParam, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}

template<typename op>
bool processData(astra::CData2D *out1, astra::CData2D *out2, float fParam1, float fParam2, const Stream &stream)
{
	assert(out1->getShape() == out2->getShape());
	CDataGPU *out1s = dynamic_cast<CDataGPU*>(out1->getStorage());
	assert(out1s);
	assert(!out1s->getArray());
	CDataGPU *out2s = dynamic_cast<CDataGPU*>(out2->getStorage());
	assert(out2s);
	assert(!out2s->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out1->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut1 = (float*)out1s->getPtr().ptr;
	float *pfOut2 = (float*)out2s->getPtr().ptr;
	unsigned int outPitch = out1s->getPtr().pitch / sizeof(float);
	assert(out1s->getPtr().pitch == out2s->getPtr().pitch);

	devFFtoDD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut1, pfOut2, fParam1, fParam2, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}


template<typename op>
bool processData(astra::CData2D *out, const astra::CData2D *in, const Stream &stream)
{
	assert(out->getShape() == in->getShape());
	CDataGPU *outs = dynamic_cast<CDataGPU*>(out->getStorage());
	assert(outs);
	assert(!outs->getArray());
	const astraCUDA::CDataGPU *ins = dynamic_cast<const astraCUDA::CDataGPU*>(in->getStorage());
	assert(ins);
	assert(!ins->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut = (float*)outs->getPtr().ptr;
	float *pfIn = (float*)ins->getPtr().ptr;
	unsigned int outPitch = outs->getPtr().pitch / sizeof(float);
	assert(outs->getPtr().pitch == ins->getPtr().pitch);

	devDtoD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut, pfIn, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}

template<typename op>
bool processData(astra::CData2D *out, const astra::CData2D *in, float fParam, const Stream &stream)
{
	assert(out->getShape() == in->getShape());
	CDataGPU *outs = dynamic_cast<CDataGPU*>(out->getStorage());
	assert(outs);
	assert(!outs->getArray());
	const astraCUDA::CDataGPU *ins = dynamic_cast<const astraCUDA::CDataGPU*>(in->getStorage());
	assert(ins);
	assert(!ins->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut = (float*)outs->getPtr().ptr;
	float *pfIn = (float*)ins->getPtr().ptr;
	unsigned int outPitch = outs->getPtr().pitch / sizeof(float);
	assert(outs->getPtr().pitch == ins->getPtr().pitch);

	devDFtoD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut, pfIn, fParam, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}

template<typename op>
bool processData(astra::CData2D *out, const astra::CData2D *in1, const astra::CData2D *in2, float fParam, const Stream &stream)
{
	assert(out->getShape() == in1->getShape());
	assert(out->getShape() == in2->getShape());
	CDataGPU *outs = dynamic_cast<CDataGPU*>(out->getStorage());
	assert(outs);
	assert(!outs->getArray());
	const astraCUDA::CDataGPU *in1s = dynamic_cast<const astraCUDA::CDataGPU*>(in1->getStorage());
	assert(in1s);
	assert(!in1s->getArray());
	const astraCUDA::CDataGPU *in2s = dynamic_cast<const astraCUDA::CDataGPU*>(in2->getStorage());
	assert(in2s);
	assert(!in2s->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut = (float*)outs->getPtr().ptr;
	float *pfIn1 = (float*)in1s->getPtr().ptr;
	float *pfIn2 = (float*)in2s->getPtr().ptr;
	unsigned int outPitch = outs->getPtr().pitch / sizeof(float);
	assert(outs->getPtr().pitch == in1s->getPtr().pitch);
	assert(outs->getPtr().pitch == in2s->getPtr().pitch);

	devDDFtoD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut, pfIn1, pfIn2, fParam, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}

template<typename op>
bool processData(astra::CData2D *out, const astra::CData2D *in1, const astra::CData2D *in2, const Stream &stream)
{
	assert(out->getShape() == in1->getShape());
	assert(out->getShape() == in2->getShape());
	CDataGPU *outs = dynamic_cast<CDataGPU*>(out->getStorage());
	assert(outs);
	assert(!outs->getArray());
	const astraCUDA::CDataGPU *in1s = dynamic_cast<const astraCUDA::CDataGPU*>(in1->getStorage());
	assert(in1s);
	assert(!in1s->getArray());
	const astraCUDA::CDataGPU *in2s = dynamic_cast<const astraCUDA::CDataGPU*>(in2->getStorage());
	assert(in2s);
	assert(!in2s->getArray());
	assert(stream.isValid());

	std::array<int, 2> dims = out->getShape();

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, (dims[1]+511)/512);
	float *pfOut = (float*)outs->getPtr().ptr;
	float *pfIn1 = (float*)in1s->getPtr().ptr;
	float *pfIn2 = (float*)in2s->getPtr().ptr;
	unsigned int outPitch = outs->getPtr().pitch / sizeof(float);
	assert(outs->getPtr().pitch == in1s->getPtr().pitch);
	assert(outs->getPtr().pitch == in2s->getPtr().pitch);

	devDDtoD<op, 32><<<gridSize, blockSize, 0, **stream>>>(pfOut, pfIn1, pfIn2, outPitch, dims[0], dims[1]);

	return stream.syncIfAuto(__FUNCTION__);
}













#define INST_DFtoD(name) \
  template bool processData<name>(astra::CData2D* out, const astra::CData2D* in, float param, const Stream& stream);

#define INST_DtoD(name) \
  template bool processData<name>(astra::CData2D* out, const astra::CData2D* in, const Stream& stream);

#define INST_DDtoD(name) \
  template bool processData<name>(astra::CData2D* out, const astra::CData2D* in1, const astra::CData2D* in2, const Stream& stream);

#define INST_DDFtoD(name) \
  template bool processData<name>(astra::CData2D* out, const astra::CData2D* in1, const astra::CData2D* in2, float fParam, const Stream& stream);

#define INST_toD(name) \
  template bool processData<name>(astra::CData2D* out, const Stream& stream);

#define INST_FtoD(name) \
  template bool processData<name>(astra::CData2D* out, float param, const Stream& stream);

#define INST_FFtoDD(name) \
  template bool processData<name>(astra::CData2D* out1, astra::CData2D* out2, float fParam1, float fParam2, const Stream& stream);



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
