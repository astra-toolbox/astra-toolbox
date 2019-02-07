// Copyright (c) 2019 Intel Corporation
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ASTRA_INC_SIMDHEADERS
#define ASTRA_INC_SIMDHEADERS

#ifdef ENABLE_SIMD
//x86 architecture
#if defined(__x86_64__) || defined(__ICC) || defined(_M_AMD64)
    #ifdef __GNUC__
        //gcc compiler
        #include <x86intrin.h>
        #include <avx512fintrin.h>
        #define ALIGN(x) __attribute__((aligned(x)))
    #endif
    #if defined(__ICC) || defined(__ICL)
        //Intel compiler
        #include <immintrin.h>
        #include <zmmintrin.h>
        #define ALIGN(x) __attribute__((aligned(x)))
    #elif defined(_MSC_VER)
        //Microsoft compiler
        #include <intrin.h>
        #if _MSC_VER > 1900
            //VS2017 or later
            #include <zmmintrin.h>
        #else
            #define NO_AVX512
        #endif
        #define ALIGN(x) __declspec(align(x))
    #endif
#endif
#endif

namespace astra
{
struct VerticalHelper;
struct HorizontalHelper;

struct GlobalParameters;
struct AngleParameters;

class DefaultFPPolicy;
class DefaultBPPolicy;

namespace Simd
{
enum VectorizationType
{
    Vt_None,
    Vt_Avx2,
    Vt_Avx512,
};
extern const VectorizationType g_VtType;
template<VectorizationType Type> struct VHelper;

struct Avx2
{
static int ProjectParallelBeamLine(DefaultFPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultBPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultFPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultBPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
};

struct Avx512
{
static int ProjectParallelBeamLine(DefaultFPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultBPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultFPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultBPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
};

}
}

#endif
