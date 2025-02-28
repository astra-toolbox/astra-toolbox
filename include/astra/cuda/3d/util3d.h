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

#ifndef _CUDA_UTIL3D_H
#define _CUDA_UTIL3D_H

#include <optional>

#include "dims3d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../2d/util.h"

namespace astraCUDA3d {

using astraCUDA::checkCuda;
using astraCUDA::TransferConstantsBuffer_t;
using astraCUDA::StreamHelper;

cudaPitchedPtr allocateVolumeData(const SDimensions3D& dims);
cudaPitchedPtr allocateProjectionData(const SDimensions3D& dims);
bool zeroVolumeData(cudaPitchedPtr& D_data, const SDimensions3D& dims);
bool zeroProjectionData(cudaPitchedPtr& D_data, const SDimensions3D& dims);
bool copyVolumeToDevice(const float* data, cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch = 0);
bool copyProjectionsToDevice(const float* data, cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch = 0);
bool copyVolumeFromDevice(float* data, const cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch = 0);
bool copyProjectionsFromDevice(float* data, const cudaPitchedPtr& D_data, const SDimensions3D& dims, unsigned int pitch = 0);
bool duplicateVolumeData(cudaPitchedPtr& D_dest, const cudaPitchedPtr& D_src, const SDimensions3D& dims); 
bool duplicateProjectionData(cudaPitchedPtr& D_dest, const cudaPitchedPtr& D_src, const SDimensions3D& dims); 


bool transferProjectionsToArray(cudaPitchedPtr D_projData, cudaArray* array, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
bool transferHostProjectionsToArray(const float *projData, cudaArray* array, const SDimensions3D& dims);
bool transferVolumeToArray(cudaPitchedPtr D_volumeData, cudaArray* array, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
bool zeroProjectionArray(cudaArray* array, const SDimensions3D& dims);
bool zeroVolumeArray(cudaArray* array, const SDimensions3D& dims);
cudaArray* allocateProjectionArray(const SDimensions3D& dims);
cudaArray* allocateVolumeArray(const SDimensions3D& dims);

bool createTextureObject3D(cudaArray* array, cudaTextureObject_t& texObj);

float dotProduct3D(cudaPitchedPtr data, unsigned int x, unsigned int y, unsigned int z);

int calcNextPowerOfTwo(int _iValue);

using astra::Vec3;
using astra::det3x;
using astra::det3y;
using astra::det3z;
using astra::det3;
using astra::cross3;

inline Vec3 scaled_cross3(const Vec3 &a, const Vec3 &b, const Vec3 &sc) {
	Vec3 ret = cross3(a, b);
	ret.x *= sc.y * sc.z;
	ret.y *= sc.x * sc.z;
	ret.z *= sc.x * sc.y;
	return ret;
}

}

#endif
