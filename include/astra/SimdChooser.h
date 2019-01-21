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

#ifndef ASTRA_INC_SIMDCHOOSER
#define ASTRA_INC_SIMDCHOOSER
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
struct Chooser
{
static int ProjectParallelBeamLine(DefaultFPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultBPPolicy& p, VerticalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultFPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
static int ProjectParallelBeamLine(DefaultBPPolicy& p, HorizontalHelper const& helper, GlobalParameters const& gp, AngleParameters const& ap);
};
}

}
#endif