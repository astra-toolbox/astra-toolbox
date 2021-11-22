/*
-----------------------------------------------------------------------
Copyright: 2010-2021, imec Vision Lab, University of Antwerp
           2014-2021, CWI, Amsterdam

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

#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H

#include <cuda.h>
#include <driver_types.h>
#include <string>

#include "astra/Globals.h"

#include "dims.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace astraCUDA {

bool copyVolumeToDevice(const float* in_data, unsigned int in_pitch,
		const SDimensions& dims,
		float* outD_data, unsigned int out_pitch);
bool copyVolumeFromDevice(float* out_data, unsigned int out_pitch,
		const SDimensions& dims,
		float* inD_data, unsigned int in_pitch);
bool copySinogramFromDevice(float* out_data, unsigned int out_pitch,
		const SDimensions& dims,
		float* inD_data, unsigned int in_pitch);
bool copySinogramToDevice(const float* in_data, unsigned int in_pitch,
		const SDimensions& dims,
		float* outD_data, unsigned int out_pitch);

bool allocateVolume(float*& D_ptr, unsigned int width, unsigned int height, unsigned int& pitch);
bool zeroVolume(float* D_data, unsigned int pitch, unsigned int width, unsigned int height);

bool allocateVolumeData(float*& D_ptr, unsigned int& pitch, const SDimensions& dims);
bool allocateProjectionData(float*& D_ptr, unsigned int& pitch, const SDimensions& dims);
bool zeroVolumeData(float* D_ptr, unsigned int pitch, const SDimensions& dims);
bool zeroProjectionData(float* D_ptr, unsigned int pitch, const SDimensions& dims);

void duplicateVolumeData(float* D_dst, float* D_src, unsigned int pitch, const SDimensions& dims);
void duplicateProjectionData(float* D_dst, float* D_src, unsigned int pitch, const SDimensions& dims);

bool createArrayAndTextureObject2D(float* data, cudaArray*& dataArray, cudaTextureObject_t& texObj, unsigned int pitch, unsigned int width, unsigned int height);
bool createTextureObjectPitch2D(float* data, cudaTextureObject_t& texObj, unsigned int pitch, unsigned int width, unsigned int height, cudaTextureAddressMode mode = cudaAddressModeBorder);


bool checkCuda(cudaError_t err, const char *msg);

float dotProduct2D(float* D_data, unsigned int pitch,
                   unsigned int width, unsigned int height);

}

#endif
