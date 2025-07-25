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

#ifndef _CUDA_ARITH3D_H
#define _CUDA_ARITH3D_H

#include <optional>

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

template<typename op> bool processVol3D(cudaPitchedPtr& out, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});

template<typename op> bool processSino3D(cudaPitchedPtr& out, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino3D(cudaPitchedPtr& out, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, float fParam, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino3D(cudaPitchedPtr& out, const cudaPitchedPtr& in1, const cudaPitchedPtr& in2, const SDimensions3D& dims, std::optional<cudaStream_t> _stream = {});



}

#endif
