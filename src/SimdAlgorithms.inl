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

#include "astra/SimdAlgorithms.h"
#include "astra/ParallelBeamLineKernelProjector2D.h"

#include "astra/DataProjectorPolicies.h"
using namespace std;
using namespace astra;
#include "astra/ParallelBeamLineKernelProjector2D.inl"

#include "astra/DataProjectorPolicies.h"
#include "astra/DataProjectorPolicies.inl"
#include <limits>
#include <algorithm>

#ifdef COMPILING_AVX2
template<> struct astra::Simd::VHelper<astra::Simd::Vt_Avx2>
{
    typedef __m256 vfloat;
    typedef __m256i vint;
    static const unsigned int Width = 8;

    static vfloat fzero() { return _mm256_setzero_ps(); }
    static vint set(int v) { return _mm256_set1_epi32(v); }
    static vfloat set(float v) { return _mm256_set1_ps(v); }
    static vfloat load(const float* pSrc) { return _mm256_load_ps(pSrc); }
    static vint load(const int* pSrc) { return _mm256_load_si256((vint*) pSrc); }
    static void store(float* pData, vfloat source) { _mm256_store_ps(pData, source); }
    static void store(int* pData, vint source) { _mm256_store_si256((vint*) pData, source); }

    static vint cvt(vfloat v) { return _mm256_cvttps_epi32(v); }
    static vfloat cvt(vint v) { return _mm256_cvtepi32_ps(v); }
    static bool is_non_zero(vfloat v) { return _mm256_testz_si256(cast(v), cast(v)) == 0; }
    static vint cmp_zero(vint v) { return _mm256_cmpeq_epi32(v, _mm256_setzero_si256()); }
    static vint bxor(vint a, vint b) { return _mm256_xor_si256(a, b); }
    static vfloat band(vfloat a, vfloat b) { return _mm256_and_ps(a, b); }

    static vfloat cast(vint v) { return _mm256_castsi256_ps(v); }
    static vint cast(vfloat v) { return _mm256_castps_si256(v); }

    static vfloat fmadd(vfloat a, vfloat b, vfloat c) { return _mm256_fmadd_ps(a, b, c); }
    static vfloat fmsub(vfloat a, vfloat b, vfloat c) { return _mm256_fmsub_ps(a, b, c); }
    static vfloat mul(vfloat a, vfloat b) { return _mm256_mul_ps(a, b); }
    static vint mul(vint a, vint b) { return _mm256_mullo_epi32(a, b); }
    static vfloat add(vfloat a, vfloat b) { return _mm256_add_ps(a, b); }
    static vint add(vint a, vint b) { return _mm256_add_epi32(a, b); }
    static vfloat sub(vfloat a, vfloat b) { return _mm256_sub_ps(a, b); }
    static vint sub(vint a, vint b) { return _mm256_sub_epi32(a, b); }

    static void zeroupper() { _mm256_zeroupper(); }
};
#endif

#ifdef COMPILING_AVX512
template<> struct astra::Simd::VHelper<astra::Simd::Vt_Avx512>
{
    typedef __m512 vfloat;
    typedef __m512i vint;
    static const unsigned int Width = 16;

    static vfloat fzero() { return _mm512_setzero_ps(); }
    static vint set(int v) { return _mm512_set1_epi32(v); }
    static vfloat set(float v) { return _mm512_set1_ps(v); }
    static vfloat load(const float* pSrc) { return _mm512_load_ps(pSrc); }
    static vint load(const int* pSrc) { return _mm512_load_epi32(pSrc); }
    static void store(float* pData, vfloat source) { _mm512_store_ps(pData, source); }
    static void store(int* pData, vint source) { _mm512_store_si512(pData, source); }

    static vint cvt(vfloat v) { return _mm512_cvttps_epi32(v); }
    static vfloat cvt(vint v) { return _mm512_cvtepi32_ps(v); }
    static bool is_non_zero(vfloat v) { return _mm512_test_epi32_mask(cast(v), cast(v)) != 0; }
    static vint cmp_zero(vint v)
    {
        //AVX512 doesn't have the same functionality as AVX2 for cmpeq instruction, so generate additional instructions to match
        const auto mask = _mm512_cmpeq_epi32_mask(v, _mm512_setzero_epi32());
        return _mm512_maskz_mov_epi32(mask, set(~0));
    }
    static vint bxor(vint a, vint b) { return _mm512_xor_epi32(a, b); }
    static vfloat band(vfloat a, vfloat b) { return cast(_mm512_and_epi32(cast(a), cast(b))); }

    static vfloat cast(vint v) { return _mm512_castsi512_ps(v); }
    static vint cast(vfloat v) { return _mm512_castps_si512(v); }

    static vfloat fmadd(vfloat a, vfloat b, vfloat c) { return _mm512_fmadd_ps(a, b, c); }
    static vfloat fmsub(vfloat a, vfloat b, vfloat c) { return _mm512_fmsub_ps(a, b, c); }
    static vfloat mul(vfloat a, vfloat b) { return _mm512_mul_ps(a, b); }
    static vint mul(vint a, vint b) { return _mm512_mullo_epi32(a, b); }
    static vfloat add(vfloat a, vfloat b) { return _mm512_add_ps(a, b); }
    static vint add(vint a, vint b) { return _mm512_add_epi32(a, b); }
    static vfloat sub(vfloat a, vfloat b) { return _mm512_sub_ps(a, b); }
    static vint sub(vint a, vint b) { return _mm512_sub_epi32(a, b); }

    //Not sure if needed for AVX512
    static void zeroupper() { _mm256_zeroupper(); }
};
#endif

namespace astra
{
namespace Simd
{
template<VectorizationType Type>
struct ParallelBeamLine
{
    typedef VHelper<Type> V;

    struct alignas(typename V::vfloat) BlockData
    {
        typename V::vfloat prevWeight;
        typename V::vfloat thisWeight;
        typename V::vfloat nextWeight;
        int indexSize;
        int thisVolumeIndex[V::Width];
    };

    struct ProjectHelper
    {
        //Front projection functions
        inline void GetInitialValue(DefaultFPPolicy&, const int[])
        {
            projectionData = V::fzero();
        }

        inline void StoreFinalValue(DefaultFPPolicy& p, const int iRayIndices[])
        {
            //Pull the values out of the register and store them into their locations
            alignas(typename V::vfloat) float32 results[V::Width];
            V::store(results, projectionData);
            for(auto i = 0u; i < V::Width; ++i) { p.addProjectionData(iRayIndices[i], results[i]); }
            V::zeroupper();
        }

        FORCEINLINE void ProjectBlock(DefaultFPPolicy& p, BlockData const& block)
        {
            //Don't worry about initializing blockValues as unused values are multiplied by 0
            alignas(typename V::vfloat) float32 blockValues[V::Width];
            alignas(typename V::vfloat) float32 weightData[V::Width];
            //if prevWeight != 0, then we need to add at least one weight from the previous index
            if(V::is_non_zero(block.prevWeight)) {
                V::store(weightData, block.prevWeight);
                for(auto i = 0u; i < V::Width; ++i) { if(weightData[i] != 0) { blockValues[i] = p.getVolumeData(block.thisVolumeIndex[i] - block.indexSize); } }
                projectionData = V::fmadd(V::load(blockValues), block.prevWeight, projectionData);
            }

            //Note that "this" will always have some value (it is either left, right, or middle)
            for(auto i = 0u; i < V::Width; ++i) { blockValues[i] = p.getVolumeData(block.thisVolumeIndex[i]); }
            projectionData = V::fmadd(V::load(blockValues), block.thisWeight, projectionData);

            //if nextWeight != 0, then we need to add at least one weight from the next index
            if(V::is_non_zero(block.nextWeight)) {
                V::store(weightData, block.nextWeight);
                for(auto i = 0u; i < V::Width; ++i) { if(weightData[i] != 0) { blockValues[i] = p.getVolumeData(block.thisVolumeIndex[i] + block.indexSize); } }
                projectionData = V::fmadd(V::load(blockValues), block.nextWeight, projectionData);
            }
        }

        //Back projection functions
        inline void GetInitialValue(DefaultBPPolicy& p, const int iRayIndices[])
        {
            //Load the projection data from the ray location
            float32 tempData[V::Width];
            for(auto i = 0u; i < V::Width; ++i) { tempData[i] = p.getProjectionData(iRayIndices[i]); }
            projectionData = V::load(tempData);
        }

        inline void StoreFinalValue(DefaultBPPolicy&, const int[])
        {
            V::zeroupper();
        }

        FORCEINLINE void ProjectBlock(DefaultBPPolicy& p, BlockData const& block)
        {
            alignas(typename V::vfloat) float32 blockValues[V::Width];
            alignas(typename V::vfloat) float32 weightData[V::Width];
            //if prevWeight != 0, then we need to add at least one weight from the previous index
            if(V::is_non_zero(block.prevWeight)) {
                V::store(blockValues, V::mul(block.prevWeight, projectionData));
                V::store(weightData, block.prevWeight);
                for(auto i = 0u; i < V::Width; ++i) { if(weightData[i] != 0) { p.addToVolumeData(block.thisVolumeIndex[i] - block.indexSize, blockValues[i]); } }
            }

            //Note that "this" will always have some value (it is either left, right, or middle)
            V::store(blockValues, V::mul(block.thisWeight, projectionData));
            for(auto i = 0u; i < V::Width; ++i) { p.addToVolumeData(block.thisVolumeIndex[i], blockValues[i]); }

            //if nextWeight != 0, then we need to add at least one weight from the next index
            if(V::is_non_zero(block.nextWeight)) {
                V::store(blockValues, V::mul(block.nextWeight, projectionData));
                V::store(weightData, block.nextWeight);
                for(auto i = 0u; i < V::Width; ++i) { if(weightData[i] != 0) { p.addToVolumeData(block.thisVolumeIndex[i] + block.indexSize, blockValues[i]); } }
            }
        }

        typename V::vfloat projectionData;
    };

    struct BlockProjectionData
    {
        alignas(typename V::vint) int as[V::Width];
        alignas(typename V::vfloat) float32 b0s[V::Width];
        alignas(typename V::vint) int iRayIndices[V::Width];
        int iterations;
        int iterationSize;
        int aSize;
        int bSize;
    };

    //This function projects a group of rays at the same angle starting at the same rank (but differing perpendicular ranks)
    //It can also project a single ray in blocks if the BlockProjectionData is properly arranged
    template<typename Policy>
    static inline void ProjectRays_Block(Policy& p, BlockProjectionData const& blockInfo, ProjectionData const& data)
    {
        BlockData block;
        block.indexSize = blockInfo.bSize;

        const auto half = V::set(0.5f);
        //a is an index, so it is an integer
        //However, every loop iteration it needs to be converted back and forth,
        //and the conversion to float is more on the critical path
        //The index should be small enough that the floating point won't lose precision (up to 6 digits)
        //More advanced AVX512 instruction sets may allow so many conversions between float and int in the future
        auto a = V::cvt(V::load(blockInfo.as));
        const auto iterationSize = V::set(static_cast<float>(blockInfo.iterationSize));
        const auto aSize = V::set(blockInfo.aSize);
        const auto bSize = V::set(blockInfo.bSize);

        //Delta matches for all rays since they are the same angle
        const auto delta = V::set(data.delta);
        //Each ray may start at a different location, giving them a different b0
        const auto b0 = V::load(blockInfo.b0s);
        //S matches for all rays since it is based on the angle of the ray
        const auto Sp1 = V::add(V::set(data.S), V::set(1.0f));

        //S and T match for all rays, meaning all the following match for all rays
        const auto lengthPerRank = V::set(data.lengthPerRank);
        const auto invTminSTimesLengthPerRank = V::set(data.invTminSTimesLengthPerRank);
        const auto invTminSTimesLengthPerRankTimesT = V::set(data.invTminSTimesLengthPerRankTimesT);
        const auto invTminSTimesLengthPerRankTimesS = V::set(data.invTminSTimesLengthPerRankTimesS);

        ProjectHelper blockHelper;
        blockHelper.GetInitialValue(p, blockInfo.iRayIndices);

        for(auto i = 0; i < blockInfo.iterations; ++i) {
            //bf = (delta * index) + b0;
            const auto bf = V::fmadd(delta, a, b0);
            //b = floor(bf + 0.5f);
            const auto b = V::cvt(V::add(bf, half));
            //offset = bf - float(b);
            const auto offset = V::sub(bf, V::cvt(b));

            //The following relation is true: -1 < offset < 1 && 0 < S <= 0.5
            //Thus: -1 < offset + S < 1.5 && -1.5 < offset - S < 1
            //left = offset < -s -> -1 < offset + (s + 1.0f) < 1.0f -> trunc(offset + (s + 1.0f)) == 0 if offset < s
            //When we truncate the result of the math, it will either be 0 if offset is less than S or not 0
            //If we then compare against 0, all bits will be set to 1 if offset is less than S
            const auto left = V::cmp_zero(V::cvt(V::add(offset, Sp1)));
            //right = offset >= s -> 1 > offset - s - 1.0f >= -1.0f -> trunc(offset - (s + 1.0f)) == 0 if offset > S
            //When we truncate the result of the math, it will either be negative if offset is greater than S, otherwise it will be 0
            //If we then compare against 0, all bits will be set to 1 if offset greater than S
            const auto right = V::cmp_zero(V::cvt(V::sub(offset, Sp1)));
            //right and left will never be both set
            //So, right ^ left = 1 if right | left, and right ^ left = 0 if !right & !left
            //Thus, center = ~(right ^ left) (could also use "or", both have same performance)
            const auto center = V::cmp_zero(V::bxor(left, right));

            //Calculate our volume indices
            V::store(block.thisVolumeIndex, V::add(V::mul(V::cvt(a), aSize), V::mul(b, bSize)));

            //preWeight = (offset + T) * invTminSTimesLengthPerRank = (offset * invTminSTimesLengthPerRank) + invTminSTimesLengthPerRankTimesT
            //Additionally mask it with "left", so that if it is to the left, it is original, otherwise 0
            const auto preWeight = V::band(V::fmadd(offset, invTminSTimesLengthPerRank, invTminSTimesLengthPerRankTimesT), V::cast(left));
            //invPreWeight = (lengthPerRank - preWeight) & left
            block.prevWeight = V::band(V::sub(lengthPerRank, preWeight), V::cast(left));

            //postWeight = (offset - S) * invTminSTimesLengthPerRank = (offset * invTminSTimesLengthPerRank) - invTminSTimesLengthPerRankTimesS
            //Additionally mask it with "right", so that if it is to the left, it is original, otherwise 0
            block.nextWeight = V::band(V::fmsub(offset, invTminSTimesLengthPerRank, invTminSTimesLengthPerRankTimesS), V::cast(right));
            //invPostWeight = (lengthPerRank - postWeight) & right
            const auto invPostWeight = V::band(V::sub(lengthPerRank, block.nextWeight), V::cast(right));
            //centerWeight = (lengthPerRank & center)
            const auto centerWeight = V::band(lengthPerRank, V::cast(center));
            //thisWeight = preWeight + invPostWeight centerWeight
            block.thisWeight = V::add(V::add(centerWeight, invPostWeight), preWeight);

            blockHelper.ProjectBlock(p, block);
            a = V::add(a, iterationSize);
        }

        blockHelper.StoreFinalValue(p, blockInfo.iRayIndices);
    }

    //This handles all Front Projections as well as BackProjections that are mostly vertical
    //This gives the best memory access pattern for these cases
    template<typename Policy, typename Helper>
    static int ProjectBlockedRange(Policy&p, Helper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
    {
        //Must first find the a ray that is in bounds, and then start there
        auto iDetector = gp.detStart;
        auto isin = false;
        do {
            const auto Dx = ap.proj->fDetSX + (iDetector + 0.5f) * ap.proj->fDetUX;
            const auto Dy = ap.proj->fDetSY + (iDetector + 0.5f) * ap.proj->fDetUY;
            const auto b0 = helper.GetB0(gp, ap, Dx, Dy);
            const auto bounds = helper.GetBounds(gp, ap, b0);
            isin = (bounds.StartStep != bounds.EndPost);
            ++iDetector;
        } while(iDetector < gp.detEnd && !isin);
        //In the off case nothing is in bound, return now
        if(!isin) { return gp.detEnd; }
        //Decrement detector because loop increments it too many times
        --iDetector;

        //Calculate number of iterations based off where the first "in" detector is
        const auto iterations = (gp.detEnd - iDetector) / V::Width;
        const auto endIteration = static_cast<int>(iDetector + iterations * V::Width);
        BlockProjectionData blockInfo;
        blockInfo.iterationSize = 1;
        std::tie(blockInfo.aSize, blockInfo.bSize) = helper.GetPixelSizes();
        ProjectionData projections[V::Width];
        KernelBounds bounds[V::Width];

        //Run through each block of detectors and project the block
        for(; iDetector < endIteration; iDetector += V::Width) {
            auto firstCommon = 0;
            auto lastCommon = std::numeric_limits<int>::max();

            //Get data for each ray we're going to project in this group
            for(auto z = 0u; z < V::Width; ++z) {
                const auto iRayIndex = blockInfo.iRayIndices[z] = ap.iAngle * gp.detCount + iDetector + z;
                p.rayPrior(iRayIndex);

                const auto Dx = ap.proj->fDetSX + (iDetector + z + 0.5f) * ap.proj->fDetUX;
                const auto Dy = ap.proj->fDetSY + (iDetector + z + 0.5f) * ap.proj->fDetUY;
                blockInfo.b0s[z] = helper.GetB0(gp, ap, Dx, Dy);
                bounds[z] = helper.GetBounds(gp, ap, blockInfo.b0s[z]);
                //If we find a ray out of bounds, return early, returning the last fully projected block
                if(bounds[z].StartStep == bounds[z].EndPost) { return iDetector; }

                projections[z] = helper.GetProjectionData(gp, ap, iRayIndex, blockInfo.b0s[z]);
                firstCommon = std::max(bounds[z].EndPre, firstCommon);
                lastCommon = std::min(bounds[z].EndMain, lastCommon);
            }

            //Project the pre and post parts of each ray, plus up to the first and last common
            for(auto z = 0u; z < V::Width; ++z) {
                for(auto a = bounds[z].StartStep; a < bounds[z].EndPre; ++a) {
                    ProjectPixelChecked(p, helper, projections[z], a);
                }
                for(auto a = bounds[z].EndPre; a < firstCommon; ++a) {
                    ProjectPixel(p, helper, projections[z], a);
                }
                for(auto a = lastCommon; a < bounds[z].EndMain; ++a) {
                    ProjectPixel(p, helper, projections[z], a);
                }
                for(auto a = bounds[z].EndMain; a < bounds[z].EndPost; ++a) {
                    ProjectPixelChecked(p, helper, projections[z], a);
                }
            }

            //Finally, we can project the block of rays
            std::fill(std::begin(blockInfo.as), std::end(blockInfo.as), firstCommon);
            blockInfo.iterations = lastCommon - firstCommon;
            ProjectRays_Block(p, blockInfo, projections[0]);
        }

        //If we reach here, then return only that which we iterated
        return endIteration;
    }

    //This handles back projections that are mostly horizontal
    //This projects a single ray in blocks as opposed to multiple rays in blocks
    //This case seems to have better memory access patterns under this method
    template<>
    static int ProjectBlockedRange<DefaultBPPolicy>(DefaultBPPolicy&p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
    {
        auto isin = false;
        BlockProjectionData blockInfo;
        blockInfo.iterationSize = V::Width;
        std::tie(blockInfo.aSize, blockInfo.bSize) = helper.GetPixelSizes();

        for(auto iDetector = gp.detStart; iDetector < gp.detEnd; ++iDetector) {
            const auto iRayIndex = ap.iAngle * gp.detCount + iDetector;
            if(!p.rayPrior(iRayIndex)) continue;

            const auto Dx = ap.proj->fDetSX + (iDetector + 0.5f) * ap.proj->fDetUX;
            const auto Dy = ap.proj->fDetSY + (iDetector + 0.5f) * ap.proj->fDetUY;
            const auto b0 = helper.GetB0(gp, ap, Dx, Dy);
            const auto bounds = helper.GetBounds(gp, ap, b0);
            if(bounds.StartStep == bounds.EndPost) { if(isin) { break; } else { continue; } }
            isin = true;

            const auto data = helper.GetProjectionData(gp, ap, iRayIndex, b0);
            std::fill(std::begin(blockInfo.iRayIndices), std::end(blockInfo.iRayIndices), iRayIndex);
            std::fill(std::begin(blockInfo.b0s), std::end(blockInfo.b0s), data.b0);
            blockInfo.iterations = (bounds.EndMain - bounds.EndPre) / V::Width;
            for(auto i = 0u; i < V::Width; ++i) { blockInfo.as[i] = bounds.EndPre + i; }

            for(auto a = bounds.StartStep; a < bounds.EndPre; ++a) {
                ProjectPixelChecked(p, helper, data, a);
            }
            ProjectRays_Block(p, blockInfo, data);
            for(auto a = bounds.EndPre + static_cast<int>(blockInfo.iterations * V::Width); a < bounds.EndMain; ++a) {
                ProjectPixel(p, helper, data, a);
            }
            for(auto a = bounds.EndMain; a < bounds.EndPost; ++a) {
                ProjectPixelChecked(p, helper, data, a);
            }

            // POLICY: RAY POSTERIOR
            p.rayPosterior(iRayIndex);
        }
        return gp.detEnd;
    }
};

}
}
