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

#ifndef _CUDA_ARITH_H
#define _CUDA_ARITH_H

#include <cuda.h>

namespace astraCUDA {


struct opAddScaled;
struct opScaleAndAdd;
struct opAddMulScaled;
struct opAddMul;
struct opAdd;
struct opAdd2;
struct opMul;
struct opMul2;
struct opDividedBy;
struct opInvert;
struct opSet;
struct opClampMin;
struct opClampMax;
struct opClampMinMask;
struct opClampMaxMask;
struct opSegmentAndMask;
struct opSetMaskedValues;

struct opMulMask;



enum VolType {
  SINO = 0,
  VOL = 1
};


template<typename op, VolType t> void processVolCopy(float* out, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVolCopy(float* out, float param, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVolCopy(float* out1, float* out2, float param1, float param2, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVolCopy(float* out, const float* in, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVolCopy(float* out, const float* in, float param, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVolCopy(float* out, const float* in1, const float* in2, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVolCopy(float* out, const float* in1, const float* in2, float param, unsigned int width, unsigned int height);

template<typename op, VolType t> void processVol(float* out, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(float* out, float fParam, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(float* out, const float* in, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(float* out, const float* in, float fParam, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(float* out, const float* in1, const float* in2, unsigned int pitch, unsigned int width, unsigned int height);

template<typename op> void processVol3D(cudaPitchedPtr& out, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out1, cudaPitchedPtr& out2, float fParam1, float fParam2, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims);

template<typename op> void processSino3D(cudaPitchedPtr& out, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out1, cudaPitchedPtr& out2, float fParam1, float fParam2, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims);



}

#endif
