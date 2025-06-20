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

#ifndef _CUDA_ARITH_H
#define _CUDA_ARITH_H

#include <optional>

namespace astraCUDA {


struct opAddScaled;
struct opScaleAndAdd;
struct opAddMulScaled;
struct opAddMul;
struct opAdd;
struct opAdd2;
struct opMul;
struct opDiv;
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


template<typename op> bool processVol(float* out, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol(float* out, float fParam, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol(float* out, const float* in, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol(float* out, const float* in, float fParam, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processVol(float* out, const float* in1, const float* in2, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});

template<typename op> bool processSino(float* out, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino(float* out, float fParam, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino(float* out1, float* out2, float fParam1, float fParam2, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino(float* out, const float* in, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino(float* out, const float* in, float fParam, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino(float* out, const float* in1, const float* in2, float fParam, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});
template<typename op> bool processSino(float* out, const float* in1, const float* in2, unsigned int pitch, const SDimensions& dims, std::optional<cudaStream_t> _stream = {});


}

#endif
