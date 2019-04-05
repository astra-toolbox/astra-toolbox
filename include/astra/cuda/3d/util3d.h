/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#include <cuda.h>
#include "dims3d.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "../2d/util.h"

namespace astraCUDA3d {

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


bool transferProjectionsToArray(cudaPitchedPtr D_projData, cudaArray* array, const SDimensions3D& dims);
bool transferHostProjectionsToArray(const float *projData, cudaArray* array, const SDimensions3D& dims);
bool transferVolumeToArray(cudaPitchedPtr D_volumeData, cudaArray* array, const SDimensions3D& dims);
bool zeroProjectionArray(cudaArray* array, const SDimensions3D& dims);
bool zeroVolumeArray(cudaArray* array, const SDimensions3D& dims);
cudaArray* allocateProjectionArray(const SDimensions3D& dims);
cudaArray* allocateVolumeArray(const SDimensions3D& dims);

bool cudaTextForceKernelsCompletion();

float dotProduct3D(cudaPitchedPtr data, unsigned int x, unsigned int y, unsigned int z);

int calcNextPowerOfTwo(int _iValue);

struct Vec3 {
	double x;
	double y;
	double z;
	Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }
	Vec3 operator+(const Vec3 &b) const {
		return Vec3(x + b.x, y + b.y, z + b.z);
	}
	Vec3 operator-(const Vec3 &b) const {
		return Vec3(x - b.x, y - b.y, z - b.z);
	}
	Vec3 operator-() const {
		return Vec3(-x, -y, -z);
	}
	double norm() const {
		return sqrt(x*x + y*y + z*z);
	}
};

static double det3x(const Vec3 &b, const Vec3 &c) {
	return (b.y * c.z - b.z * c.y);
}
static double det3y(const Vec3 &b, const Vec3 &c) {
	return -(b.x * c.z - b.z * c.x);
}

static double det3z(const Vec3 &b, const Vec3 &c) {
	return (b.x * c.y - b.y * c.x);
}

static double det3(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
	return a.x * det3x(b,c) + a.y * det3y(b,c) + a.z * det3z(b,c);
}

static Vec3 cross3(const Vec3 &a, const Vec3 &b) {
	return Vec3(det3x(a,b), det3y(a,b), det3z(a,b));
}

static Vec3 scaled_cross3(const Vec3 &a, const Vec3 &b, const Vec3 &sc) {
	Vec3 ret = cross3(a, b);
	ret.x *= sc.y * sc.z;
	ret.y *= sc.x * sc.z;
	ret.z *= sc.x * sc.y;
	return ret;
}


}

#endif
