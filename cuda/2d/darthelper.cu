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
#include "darthelper.h"
#include <cassert>

namespace astraCUDA {

// CUDA function for the selection of ROI
__global__ void devRoiSelect(float* in, float radius, unsigned int pitch, unsigned int width, unsigned int height)
{
	float x = (float)(threadIdx.x + 16*blockIdx.x);
	float y = (float)(threadIdx.y + 16*blockIdx.y);

	float w = (width-1.0f)*0.5f;
	float h = (height-1.0f)*0.5f;

	if ((x-w)*(x-w) + (y-h)*(y-h) > radius * radius * 0.25f) 
	{
		float* d = (float*)in;
		unsigned int o = y*pitch+x; 
		d[o] = 0.0f;
	}
}

void roiSelect(float* out, float radius, unsigned int width, unsigned int height)
{
	float* D_data;

	unsigned int pitch;
	// We abuse dims here...
	SDimensions dims;
	dims.iVolWidth = width;
	dims.iVolHeight = height;

	allocateVolumeData(D_data, pitch, dims);
	copyVolumeToDevice(out, width, dims, D_data, pitch);

	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);
	devRoiSelect<<<gridSize, blockSize>>>(D_data, radius, pitch, width, height);

	copyVolumeFromDevice(out, width, dims, D_data, pitch);

	cudaFree(D_data);
}




// CUDA function for the masking of DART with a radius == 1
__global__ void devDartMask(float* mask, const float* in, unsigned int conn, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	unsigned int y = threadIdx.y + 16*blockIdx.y;

	// Sacrifice the border pixels to simplify the implementation. 
	if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
		float* d = (float*)in;
		float* m = (float*)mask;

		unsigned int o2 = y*pitch+x; // On this row.
		unsigned int o1 = o2 - pitch; // On previous row.
		unsigned int o3 = o2 + pitch; // On next row.

		if ((conn == 8 && // 8-connected
		        (d[o1 - 1] != d[o2] || d[o1] != d[o2] || d[o1 + 1] != d[o2] || 
		         d[o2 - 1] != d[o2] ||                   d[o2 + 1] != d[o2] ||
				 d[o3 - 1] != d[o2] || d[o3] != d[o2] || d[o3 + 1] != d[o2] ))
			|| 
			(conn == 4 && // 4-connected
		        (                      d[o1] != d[o2] ||                      
		         d[o2 - 1] != d[o2] ||                  d[o3 + 1] != d[o2] ||
				                       d[o3] != d[o2]                      )))
		{
			m[o2] = 1.0f;
		}
	}
}


// CUDA function for the masking of DART with a radius > 1
__global__ void devDartMaskRadius(float* mask, const float* in, unsigned int conn, unsigned int radius, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	unsigned int y = threadIdx.y + 16*blockIdx.y;

	// Sacrifice the border pixels to simplify the implementation. 
	if (x > radius-1 && x < width - radius && y > radius-1 && y < height - radius) 
	{
		float* d = (float*)in;
		float* m = (float*)mask;

		int r = radius;

		// o2: index of the current center pixel
		int o2 = y*pitch+x;

		if (conn == 8) // 8-connected
		{
			for (int row = -r; row <= r; row++) 
			{
				int o1 = (y+row)*pitch+x; 
				for (int col = -r; col <= r; col++) 
				{
					if (d[o1 + col] != d[o2]) {m[o2] = 1.0f; return;}
				}
			}
		}
		else if (conn == 4) // 4-connected
		{
			// horizontal
			unsigned int o1 = y*pitch+x; 
			for (int col = -r; col <= r; col++) 
			{
				if (d[o1 + col] != d[o2]) {m[o2] = 1.0f; return;}
			}

			// vertical
			for (int row = -r; row <= r; row++) 
			{
				unsigned int o1 = (y+row)*pitch+x; 
				if (d[o1] != d[o2]) {m[o2] = 1.0f; return;}
			}
		}
	}
}


// CUDA function for the masking of ADART with a radius == 1
__global__ void devADartMask(float* mask, const float* in, unsigned int conn, unsigned int threshold, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	unsigned int y = threadIdx.y + 16*blockIdx.y;

	// Sacrifice the border pixels to simplify the implementation. 
	if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
		float* d = (float*)in;
		float* m = (float*)mask;

		unsigned int o2 = y*pitch+x; // On this row.
		unsigned int o1 = o2 - pitch; // On previous row.
		unsigned int o3 = o2 + pitch; // On next row.

		if (conn == 8)
		{
			if (d[o1 - 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o1    ] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o1 + 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
		    if (d[o2 - 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o2 + 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
		    if (d[o3 - 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o3    ] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o3 + 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
		}
		else if (conn == 4)
		{
			if (d[o1    ] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
		    if (d[o2 - 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o2 + 1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			if (d[o3    ] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
		}
	}
}


// CUDA function for the masking of ADART with a radius > 1
__global__ void devADartMaskRadius(float* mask, const float* in, unsigned int conn, unsigned int radius, unsigned int threshold, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	unsigned int y = threadIdx.y + 16*blockIdx.y;

	// Sacrifice the border pixels to simplify the implementation. 
	if (x > radius-1 && x < width - radius && y > radius-1 && y < height - radius)
	{
		float* d = (float*)in;
		float* m = (float*)mask;
	
		int r = radius;

		unsigned int o2 = y*pitch+x; // On this row.

		if (conn == 8)
		{
			for (int row = -r; row <= r; row++) 
			{
				unsigned int o1 = (y+row)*pitch+x; 
				for (int col = -r; col <= r; col++) 
				{
					if (d[o1+col] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
				}
			}
		}
		else if (conn == 4)
		{
			// horizontal
			for (int col = -r; col <= r; col++) 
			{
				if (d[o2+col] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			}

			// vertical
			for (int row = -r; row <= r; row++) 
			{
				unsigned int o1 = (y+row)*pitch+x; 
				if (d[o1] != d[o2] && --threshold == 0) {m[o2] = 1.0f; return;}
			}
		}
	}
}


void dartMask(float* mask, const float* segmentation, unsigned int conn, unsigned int radius, unsigned int threshold, unsigned int width, unsigned int height)
{
	float* D_segmentationData;
	float* D_maskData;

	unsigned int pitch;
	// We abuse dims here...
	SDimensions dims;
	dims.iVolWidth = width;
	dims.iVolHeight = height;

	allocateVolumeData(D_segmentationData, pitch, dims);
	copyVolumeToDevice(segmentation, width, dims, D_segmentationData, pitch);

	allocateVolumeData(D_maskData, pitch, dims);
	zeroVolumeData(D_maskData, pitch, dims);

	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);

	if (threshold == 1 && radius == 1)
		devDartMask<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, pitch, width, height);
	else if (threshold > 1 && radius == 1)
		devADartMask<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, threshold, pitch, width, height);
	else if (threshold == 1 && radius > 1)
		devDartMaskRadius<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, radius, pitch, width, height);
	else 
		devADartMaskRadius<<<gridSize, blockSize>>>(D_maskData, D_segmentationData, conn, radius, threshold, pitch, width, height);

	copyVolumeFromDevice(mask, width, dims, D_maskData, pitch);

	cudaFree(D_segmentationData);
	cudaFree(D_maskData);

}


__global__ void devDartSmoothingRadius(float* out, const float* in, float b, unsigned int radius, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	unsigned int y = threadIdx.y + 16*blockIdx.y;

	// Sacrifice the border pixels to simplify the implementation.
	if (x > radius-1 && x < width - radius && y > radius-1 && y < height - radius)
	{
		float* d = (float*)in;
		float* m = (float*)out;

		unsigned int o2 = y*pitch+x;
		int r = radius;
                float count = 4*r*(r+1);
		float res = -d[o2];

		for (int row = -r; row <= r; row++) 
		{
			unsigned int o1 = (y+row)*pitch+x; 
			for (int col = -r; col <= r; col++) 
			{
				res += d[o1+col];
			}
		}

		res *= b / count;
		res += (1.0f-b) * d[o2];

		m[o2] = res;
	}
}


__global__ void devDartSmoothing(float* out, const float* in, float b, unsigned int pitch, unsigned int width, unsigned int height)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	unsigned int y = threadIdx.y + 16*blockIdx.y;

	// Sacrifice the border pixels to simplify the implementation. 
	if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
		float* d = (float*)in;
		float* m = (float*)out;

		unsigned int o2 = y*pitch+x; // On this row.
		unsigned int o1 = o2 - pitch; // On previous row.
		unsigned int o3 = o2 + pitch; // On next row.

		m[o2] = (1.0f-b) * d[o2] + b * 0.125f * (d[o1 - 1] + d[o1] + d[o1 + 1] + d[o2 - 1] + d[o2 + 1] + d[o3 - 1] + d[o3] + d[o3 + 1]);
	}
}


void dartSmoothing(float* out, const float* in, float b, unsigned int radius, unsigned int width, unsigned int height)
{
	float* D_inData;
	float* D_outData;

	unsigned int pitch;
	// We abuse dims here...
	SDimensions dims;
	dims.iVolWidth = width;
	dims.iVolHeight = height;

	allocateVolumeData(D_inData, pitch, dims);
	copyVolumeToDevice(in, width, dims, D_inData, pitch);

	allocateVolumeData(D_outData, pitch, dims);
	zeroVolumeData(D_outData, pitch, dims);

	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, (height+15)/16);
	if (radius == 1)
		devDartSmoothing<<<gridSize, blockSize>>>(D_outData, D_inData, b, pitch, width, height);
	else
		devDartSmoothingRadius<<<gridSize, blockSize>>>(D_outData, D_inData, b, radius, pitch, width, height);

	copyVolumeFromDevice(out, width, dims, D_outData, pitch);

	cudaFree(D_outData);
	cudaFree(D_inData);

}



_AstraExport bool setGPUIndex(int iGPUIndex)
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
