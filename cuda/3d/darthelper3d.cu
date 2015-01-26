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
#include "dims3d.h"
#include "darthelper3d.h"
#include <cassert>

namespace astraCUDA3d {


	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------
	__global__ void devDartSmoothing(cudaPitchedPtr out, cudaPitchedPtr in, float b, SDimensions3D dims)
	{
		unsigned int x = threadIdx.x + 16*blockIdx.x;
		unsigned int y = threadIdx.y + 16*blockIdx.y;

		// Sacrifice the border pixels to simplify the implementation. 
		if (x > 0 && x < dims.iVolX - 1 && y > 0 && y < dims.iVolY - 1) {
			
			float* d = (float*)in.ptr;
			float* m = (float*)out.ptr;

			unsigned int index;
			unsigned int p = (out.pitch >> 2);

			for (unsigned int z = 0; z <= dims.iVolZ-1; z++) {

				float res = 0.0f;

				// bottom slice
				if (z > 0) {
					index = ((z-1)*dims.iVolY + y) * p + x;
					res += d[index-p-1] + d[index-p] + d[index-p+1] +
						d[index  -1] + d[index  ] + d[index  +1] +
						d[index+p-1] + d[index+p] + d[index+p+1];
				}

				// top slice
				if (z < dims.iVolZ-1) {
					index = ((z+1)*dims.iVolY + y) * p + x;
					res += d[index-p-1] + d[index-p] + d[index-p+1] +
						d[index  -1] + d[index  ] + d[index  +1] +
						d[index+p-1] + d[index+p] + d[index+p+1];
				}
	
				// same slice
				index = (z*dims.iVolY + y) * p + x;
				res += d[index-p-1] + d[index-p] + d[index-p+1] +
					d[index  -1] +              d[index  +1] +
					d[index+p-1] + d[index+p] + d[index+p+1];

				// result
				m[index] = (1.0f-b) * d[index] + b * 0.038461538f * res;

			}

		}
	}

	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------
	void dartSmoothing(float* out, const float* in, float b, unsigned int radius, SDimensions3D dims)
	{
		cudaPitchedPtr D_inData;
		D_inData = allocateVolumeData(dims);
		copyVolumeToDevice(in, D_inData, dims);

		cudaPitchedPtr D_outData;
		D_outData = allocateVolumeData(dims);
		copyVolumeToDevice(out, D_outData, dims);

		dim3 blockSize(16,16);
		dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+15)/16);

		devDartSmoothing<<<gridSize, blockSize>>>(D_outData, D_inData, b, dims);

		copyVolumeFromDevice(out, D_outData, dims);

		cudaFree(D_outData.ptr);
		cudaFree(D_inData.ptr);

	}


	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// CUDA function for the masking of DART with a radius == 1
	__global__ void devDartMasking(cudaPitchedPtr mask, cudaPitchedPtr in, unsigned int conn, SDimensions3D dims)
	{
		unsigned int x = threadIdx.x + 16*blockIdx.x;
		unsigned int y = threadIdx.y + 16*blockIdx.y;

		// Sacrifice the border pixels to simplify the implementation. 
		if (x > 0 && x < dims.iVolX - 1 && y > 0 && y < dims.iVolY - 1) {
			
			float* d = (float*)in.ptr;
			float* m = (float*)mask.ptr;

			unsigned int index;
			unsigned int p = (in.pitch >> 2);

			for (unsigned int z = 0; z <= dims.iVolZ-1; z++) {
				
				unsigned int o2 = (z*dims.iVolY + y) * p + x;
				
				m[o2] = 0.0f;

				// bottom slice
				if (z > 0) {
					index = ((z-1)*dims.iVolY + y) * p + x;
					if ((conn == 26 && 
						(d[index-p-1] != d[o2] || d[index-p] != d[o2] || d[index-p+1] != d[o2] || 
						 d[index  -1] != d[o2] || d[index  ] != d[o2] || d[index  +1] != d[o2] || 
						 d[index+p-1] != d[o2] || d[index+p] != d[o2] || d[index+p+1] != d[o2] ))
						|| 
						(conn == 6 && d[index] != d[o2]))
					{
						m[o2] = 1.0f;
						continue;
					}
				}

				// top slice
				if (z < dims.iVolZ-1) {
					index = ((z+1)*dims.iVolY + y) * p + x;
					if ((conn == 26 && 
						(d[index-p-1] != d[o2] || d[index-p] != d[o2] || d[index-p+1] != d[o2] || 
						 d[index  -1] != d[o2] || d[index  ] != d[o2] || d[index  +1] != d[o2] || 
						 d[index+p-1] != d[o2] || d[index+p] != d[o2] || d[index+p+1] != d[o2] ))
						|| 
						(conn == 6 && d[index] != d[o2]))
					{
						m[o2] = 1.0f;
						continue;
					}
				}

				// other slices
				index = (z*dims.iVolY + y) * p + x;
				if ((conn == 26 && 
					(d[index-p-1] != d[o2] || d[index-p] != d[o2] || d[index-p+1] != d[o2] || 
					 d[index  -1] != d[o2] ||                        d[index  +1] != d[o2] || 
					 d[index+p-1] != d[o2] || d[index+p] != d[o2] || d[index+p+1] != d[o2] ))
					|| 
					(conn == 6 && 
					(                         d[index-p] != d[o2] || 
					 d[index  -1] != d[o2] ||                        d[index  +1] != d[o2] || 
					                          d[index+p] != d[o2]                          )))
				{
					m[o2] = 1.0f;
					continue;
				}

			}

		}
	}


	
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------
	void dartMasking(float* mask, const float* segmentation, unsigned int conn, unsigned int radius, unsigned int threshold, SDimensions3D dims)
	{
		cudaPitchedPtr D_maskData;
		D_maskData = allocateVolumeData(dims);
		copyVolumeToDevice(mask, D_maskData, dims);

		cudaPitchedPtr D_segmentationData;
		D_segmentationData = allocateVolumeData(dims);
		copyVolumeToDevice(segmentation, D_segmentationData, dims);

		dim3 blockSize(16,16);
		dim3 gridSize((dims.iVolX+15)/16, (dims.iVolY+15)/16);

		if (threshold == 1 && radius == 1)
			devDartMasking<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, dims);
		//else if (threshold > 1 && radius == 1)
		//	devADartMask<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, threshold, pitch, width, height, 1, 1);
		//else if (threshold == 1 && radius > 1)
		//	devDartMaskRadius<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, radius, pitch, width, height, 1, 1);
		//else 
		//	devADartMaskRadius<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, radius, threshold, pitch, width, height, 1, 1);

		copyVolumeFromDevice(mask, D_maskData, dims);

		cudaFree(D_maskData.ptr);
		cudaFree(D_segmentationData.ptr);

	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------

	bool setGPUIndex(int iGPUIndex)
	{
		if (iGPUIndex != -1) {
			cudaSetDevice(iGPUIndex);
			cudaError_t err = cudaGetLastError();

			// Ignore errors caused by calling cudaSetDevice multiple times
			if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
				return false;
		}

		return true;
	}


}
