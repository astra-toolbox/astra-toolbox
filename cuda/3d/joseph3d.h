/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
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

#ifndef _CUDA_JOSEPH3D_H
#define _CUDA_JOSEPH3D_H

#include <cuda.h>

namespace astraCUDA3d {

// x=0, y=1, z=2
template<float (*interpolation_method)(float,float,float,float (*)(float, float, float)),
         float (*tex_lookup_xyz)(float,float,float)>
struct DIR_X {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return x; }
	__device__ float c1(float x, float y, float z) const { return y; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float x(float f0, float f1, float f2) const { return f0; }
	__device__ float y(float f0, float f1, float f2) const { return f1; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
	__device__ static float tex_lookup(float f0, float f1, float f2) { return tex_lookup_xyz(f0, f1, f2); }
	__device__ float tex(float f0, float f1, float f2) const { return interpolation_method(f0, f1, f2, tex_lookup); }
};

// y=0, x=1, z=2
template<float (*interpolation_method)(float,float,float,float (*)(float, float, float)),
         float (*tex_lookup_xyz)(float,float,float)>
struct DIR_Y {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return y; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f0; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
	__device__ static float tex_lookup(float f0, float f1, float f2) { return tex_lookup_xyz(f1, f0, f2); }
	__device__ float tex(float f0, float f1, float f2) const { return interpolation_method(f0, f1, f2, tex_lookup); }
};

// z=0, x=1, y=2
template<float (*interpolation_method)(float,float,float,float (*)(float, float, float)),
         float (*tex_lookup_xyz)(float,float,float)>
struct DIR_Z {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float c0(float x, float y, float z) const { return z; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return y; }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f2; }
	__device__ float z(float f0, float f1, float f2) const { return f0; }
	__device__ static float tex_lookup(float f0, float f1, float f2) { return tex_lookup_xyz(f1, f2, f0); }
	__device__ float tex(float f0, float f1, float f2) const { return interpolation_method(f0, f1, f2, tex_lookup); }
};

struct SCALE_CUBE {
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1+a2*a2+1.0f) * fOutputScale; }
};

struct SCALE_NONCUBE {
	float fScale1;
	float fScale2;
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1*fScale1+a2*a2*fScale2+1.0f) * fOutputScale; }
};

}

#endif
