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

#include "astra/cuda/2d/sart.h"
#include "astra/cuda/2d/util.h"
#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Data2D.h"

#include <cstdio>
#include <cassert>
#include <array>

namespace astraCUDA {

__global__ void devMUL_SART(float* pfOut, const float* pfIn, unsigned int width)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	pfOut[x] *= pfIn[x];
}

bool mul_SART(astra::CData2D *dst, const astra::CData2D *src, int angle)
{
	assert(dst->getShape()[0] == src->getShape()[0]);
	assert(dst->getShape()[1] == 1);

	astraCUDA::CDataGPU *dsts = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(dsts);
	assert(!dsts->getArray());
	const astraCUDA::CDataGPU *srcs = dynamic_cast<const astraCUDA::CDataGPU*>(src->getStorage());
	assert(srcs);
	assert(!srcs->getArray());

	std::array<int, 2> dims = src->getShape();
	assert(angle >= 0);
	assert(angle < dims[1]);

	const float *src_ptr = (float*)srcs->getPtr().ptr;
	src_ptr += angle * (srcs->getPtr().pitch / sizeof(float));

	dim3 blockSize(16,16);
	dim3 gridSize((dims[0]+15)/16, 1);

	cudaStream_t stream;
	if (!checkCuda(cudaStreamCreate(&stream), "MUL_SART stream"))
		return false;

	devMUL_SART<<<gridSize, blockSize, 0, stream>>>((float*)dsts->getPtr().ptr, src_ptr, dims[0]);

	bool ok = checkCuda(cudaStreamSynchronize(stream), "MUL_SART");
	cudaStreamDestroy(stream);
	return ok;
}

bool copy_SART(astra::CData2D *dst, const astra::CData2D *src, int angle)
{
	assert(dst->getShape()[0] == src->getShape()[0]);
	assert(dst->getShape()[1] == 1);

	astraCUDA::CDataGPU *dsts = dynamic_cast<astraCUDA::CDataGPU*>(dst->getStorage());
	assert(dsts);
	assert(!dsts->getArray());
	const astraCUDA::CDataGPU *srcs = dynamic_cast<const astraCUDA::CDataGPU*>(src->getStorage());
	assert(srcs);
	assert(!srcs->getArray());

	std::array<int, 2> dims = src->getShape();
	assert(angle >= 0);
	assert(angle < dims[1]);

	const float *src_ptr = (float*)srcs->getPtr().ptr;
	src_ptr += angle * (srcs->getPtr().pitch / sizeof(float));

	return checkCuda(cudaMemcpy(dsts->getPtr().ptr, src_ptr, sizeof(float)*dims[0], cudaMemcpyDeviceToDevice), "copy_SART memcpy");
}

}


