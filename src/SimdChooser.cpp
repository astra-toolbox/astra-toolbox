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

#include "astra/SimdChooser.h"
#include "astra/SimdAlgorithms.h"
#include "astra/ParallelBeamLineKernelProjector2D.h"

namespace astra
{
namespace Simd
{

template<typename Policy, typename Helper>
int ChooseParallelBeamLine(Policy& p, Helper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
{
    switch(g_VtType)
    {
    case Vt_Avx512:
        //Under the 512F instruction set, there doesn't appear to be a performance gain to use 512 over 256
        //However, if the memory bound nature were to ever change, re-enabling AVX512F would potentially improve performance
        //return Avx512::ProjectParallelBeamLine(p, helper, gp, ap);
    case Vt_Avx2:
        return Avx2::ProjectParallelBeamLine(p, helper, gp, ap);
    case Vt_None:
    default:
        return gp.detStart;
    }
}

int Chooser::ProjectParallelBeamLine(DefaultFPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
{
    return ChooseParallelBeamLine(p, helper, gp, ap);
}
int Chooser::ProjectParallelBeamLine(DefaultBPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
{
    return ChooseParallelBeamLine(p, helper, gp, ap);
}
int Chooser::ProjectParallelBeamLine(DefaultFPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
{
    return ChooseParallelBeamLine(p, helper, gp, ap);
}
int Chooser::ProjectParallelBeamLine(DefaultBPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
{
    return ChooseParallelBeamLine(p, helper, gp, ap);
}

}
}
