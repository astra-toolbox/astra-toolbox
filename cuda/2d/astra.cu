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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/par_fp.h"
#include "astra/cuda/2d/fan_fp.h"
#include "astra/cuda/2d/par_bp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/arith.h"
#include "astra/cuda/2d/astra.h"
#include "astra/cuda/2d/fft.h"

// For fan beam FBP weighting
#include "astra/cuda/3d/fdk.h"

#include "astra/GeometryUtil2D.h"
#include "astra/VolumeGeometry2D.h"
#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/ParallelVecProjectionGeometry2D.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/Logging.h"

#include <cstdio>
#include <cassert>
#include <fstream>

using namespace astraCUDA;
using namespace std;


namespace astra {

enum CUDAProjectionType {
	PROJ_PARALLEL,
	PROJ_FAN
};

}

namespace astraCUDA {


_AstraExport std::string getCudaDeviceString(int device)
{
	char buf[1024];
	cudaError_t err;
	if (device == -1) {
		err = cudaGetDevice(&device);
		if (err != cudaSuccess) {
			return "Error getting current GPU index";
		}
	}

	cudaDeviceProp prop;
	err = cudaGetDeviceProperties(&prop, device);
	if (err != cudaSuccess) {
		(void)cudaGetLastError(); // Reset error condition
		snprintf(buf, 1024, "GPU #%d: Invalid device (%d): %s", device, err, cudaGetErrorString(err));
		return buf;
	}

	long mem = prop.totalGlobalMem / (1024*1024);
	snprintf(buf, 1024, "GPU #%d: %s, with %ldMB, CUDA compute capability %d.%d", device, prop.name, mem, prop.major, prop.minor);
	return buf;
}

_AstraExport bool setGPUIndex(int iGPUIndex)
{
        if (iGPUIndex != -1) {
                cudaError_t err = cudaSetDevice(iGPUIndex);

                // Ignore errors caused by calling cudaSetDevice multiple times
                if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
                        return false;
        }

        return true;
}

_AstraExport size_t availableGPUMemory()
{
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess)
        return 0;
    return free;
}




}
