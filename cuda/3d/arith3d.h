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

#ifndef _CUDA_ARITH3D_H
#define _CUDA_ARITH3D_H

#include <cuda.h>

namespace astraCUDA3d {

struct opAddScaled;
struct opScaleAndAdd;
struct opAddMulScaled;
struct opAddMul;
struct opAdd;
struct opMul;
struct opMul2;
struct opDividedBy;
struct opInvert;
struct opSet;
struct opClampMin;
struct opClampMax;

enum VolType {
  SINO = 0,
  VOL = 1
};


template<typename op, VolType t> void processVol(CUdeviceptr* out, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(CUdeviceptr* out, float fParam, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(CUdeviceptr* out, const CUdeviceptr* in, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(CUdeviceptr* out, const CUdeviceptr* in, float fParam, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(CUdeviceptr* out, const CUdeviceptr* in1, const CUdeviceptr* in2, float fParam, unsigned int pitch, unsigned int width, unsigned int height);
template<typename op, VolType t> void processVol(CUdeviceptr* out, const CUdeviceptr* in1, const CUdeviceptr* in2, unsigned int pitch, unsigned int width, unsigned int height);

template<typename op> void processVol3D(cudaPitchedPtr& out, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims);
template<typename op> void processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims);

template<typename op> void processSino3D(cudaPitchedPtr& out, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims);
template<typename op> void processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims);



}

#endif
