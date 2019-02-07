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

#include "astra/Globals.h"
#include "astra/SimdAlgorithms.h"
#ifdef ASTRA_CUDA
#include "astra/cuda/2d/astra.h"
#endif

namespace astra {

bool running_in_matlab=false;

_AstraExport bool cudaAvailable() {
#ifdef ASTRA_CUDA
	return astraCUDA::availableGPUMemory() > 0;
#else
	return false;
#endif
}

#ifdef ENABLE_SIMD
Simd::VectorizationType GetSupportedVectorization()
{
#ifdef __GNUC__
    __builtin_cpu_init();
    if(__builtin_cpu_supports("fma"))
    {
        if(__builtin_cpu_supports("avx512f")) { return Simd::Vt_Avx512; }
        if(__builtin_cpu_supports("avx2")) { return Simd::Vt_Avx2; }
    }
#endif //__GNUC__

#if defined(__ICC) or defined(__ICL)
        if(_may_i_use_cpu_feature(_FEATURE_FMA))
        {
            if(_may_i_use_cpu_feature(_FEATURE_AVX512F)) { return Simd::Vt_Avx512; }
            if(_may_i_use_cpu_feature(_FEATURE_AVX2)) { return Simd::Vt_Avx2; }
        }
#elif defined(_MSC_VER)
    int cpui[4];
        __cpuid(cpui, 0);
        const auto nids = cpui[0];
        if(nids >= 7)
        {
            __cpuidex(cpui, 1, 0);
            const auto fma = (cpui[2] & (1u << 12)) != 0;
            __cpuidex(cpui, 7, 0);
#ifndef NO_AVX512
            const auto avx512 = (cpui[1] & (1u << 16)) != 0;
            if(avx512 && fma) { return Simd::Vt_Avx512; }
#endif
            const auto avx2 = (cpui[1] & (1u << 5)) != 0;
            if(avx2 && fma) { return Simd::Vt_Avx2; }
        }
#endif

    return Simd::Vt_None;
}

const Simd::VectorizationType Simd::g_VtType = GetSupportedVectorization();
#else
const Simd::VectorizationType Simd::g_VtType = Simd::Vt_None;
#endif

}

