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

#include "astra/cuda/stream_internal.h"

namespace astraCUDA3d {

// Check that we don't allocate more than CompositeGeometryManager expects
static_assert(g_blockSlices <= g_cuda3d_blockSlices);


// Combined "outer loop" for calling ray-driven kernels. It sorts projections
// per major direction (x, y, z), and permutes axes and coordinates so the
// kernels always project along the z axis.
// Typical use is the forward projection (mode == MODE_FORWARD). In this case,
// it permutes the input data into a temp buffer if necessary, binds it to a
// layered texture, and calls the kernel dispatch function.
// In MODE_FW_NOINPUT, the input data is skipped, for kernels that require no
// input; in particular SumSqW.
// In MODE_BACKWARD, the temporary buffer is still for the volume data, but
// that is then output. This is for backward ray-driven kernels; in particular
// matching kernels to our default forward kernels.
template<typename TCBUF, typename PROJ>
bool RayDrivenProjector3D<TCBUF,PROJ>::project(cudaPitchedPtr D_volumeData,
                                               cudaPitchedPtr D_projData,
                                               RayDrivenProjector3D::EMode mode,
                                               const SDimensions3D& dims,
                                               const std::vector<PROJ> &angles,
                                               const SProjectorParams3D& params,
                                               const Stream& stream)
{
	if (!stream.isValid())
		return false;

	TCBUF tcbuf(g_MaxAngles);

	std::vector<PROJ> anglesX, anglesY, anglesZ;
	std::vector<int> indicesX, indicesY, indicesZ;

	// Sort projections by major axis
	bool ok = sortProjections(dims, angles,
	                          anglesX, anglesY, anglesZ,
	                          indicesX, indicesY, indicesZ);
	if (!ok)
		return false;

	for (int majordim = 0; majordim < 3; ++majordim) {
		const std::vector<PROJ>& anglesDim = (majordim == 0) ? anglesX : ( (majordim == 1) ? anglesY : anglesZ );
		const std::vector<int>& indicesDim = (majordim == 0) ? indicesX : ( (majordim == 1) ? indicesY : indicesZ );

		assert(indicesDim.size() == anglesDim.size());

		if (anglesDim.empty())
			continue;

		SDimensions3D permutedDims = dims;
		if (majordim == 0) {
			permutedDims.iVolX = dims.iVolY;
			permutedDims.iVolY = dims.iVolZ;
			permutedDims.iVolZ = dims.iVolX;
		} else if (majordim == 1) {
			permutedDims.iVolX = dims.iVolX;
			permutedDims.iVolY = dims.iVolZ;
			permutedDims.iVolZ = dims.iVolY;
		} else if (majordim == 2) {
			// No need for permuting coordinates when projecting along Z major axis
		}

		SDimensions3D bufDims = permutedDims;
		bufDims.iVolZ = std::min(permutedDims.iVolZ, g_blockSlices);

		cudaArray* D_volArray = nullptr;
		if (mode == MODE_FORWARD) {
			D_volArray = allocateVolumeArray_LayeredZ(bufDims);
			if (!D_volArray)
				return false;
		}

		cudaTextureObject_t D_texObj;
		if (mode == MODE_FORWARD && !createTextureObject3D(D_volArray, D_texObj)) {
			logCuda(cudaFreeArray(D_volArray), "FP3D free array");
			return false;
		}

		//fprintf(stderr, "tmpVolBuffer: %d x %d x %d\n", bufDims.iVolX, bufDims.iVolY, bufDims.iVolZ);
		cudaPitchedPtr D_tmpVolBuffer;
		if (mode != MODE_FW_NOINPUT && majordim != 2) {
			// No need for a temporary buffer when projecting along Z major axis
			D_tmpVolBuffer = allocateVolumeData(bufDims);
			if (!D_tmpVolBuffer.ptr) {
				if (mode == MODE_FORWARD) {
					logCuda(cudaFreeArray(D_volArray), "FP3D free array");
					logCuda(cudaDestroyTextureObject(D_texObj), "FP3D destroy texture");
				}
				return false;
			}
		}

		for (unsigned int i = 0; i < permutedDims.iVolZ; i += g_blockSlices) {
			SDimensions3D blockDims = bufDims;
			if (i + g_blockSlices > permutedDims.iVolZ)
				blockDims.iVolZ = permutedDims.iVolZ - i;
			assert(blockDims.iVolZ <= bufDims.iVolZ);

			//fprintf(stderr, "Major dim %d block at X=%d, block dims %d x %d x %d\n", majordim, i, blockDims.iVolX, blockDims.iVolY, blockDims.iVolZ);

			if (mode == MODE_FORWARD) {
				if (majordim == 0) {
					permute_XYZ_to_ZXY_subX(D_tmpVolBuffer, D_volumeData, i, blockDims, dims, false, **stream);
				} else if (majordim == 1) {
					permute_XYZ_to_XZY_subY(D_tmpVolBuffer, D_volumeData, i, blockDims, dims, false, **stream);
				}
			}
			if (mode != MODE_FW_NOINPUT && majordim == 2) {
				// We can directly point D_tmpVolBuffer at the right slices of D_volumeData
				D_tmpVolBuffer = D_volumeData;
				D_tmpVolBuffer.ptr = (uint8_t*)(D_volumeData.ptr) + (size_t)i * dims.iVolY * D_volumeData.pitch;
			}

			if (mode == MODE_FORWARD)
				ok &= transferVolumeToArray(D_tmpVolBuffer, D_volArray, blockDims, **stream);

			if (!ok)
				break;

			// TODO: check if some modes can skip this
			ok &= stream.sync("FP3D sync");
			if (!ok)
				break;

			for (unsigned int iAngle = 0; iAngle < anglesDim.size(); iAngle += g_MaxAngles) {
				unsigned int iEndAngle = iAngle + g_MaxAngles;
				if (iEndAngle >= anglesDim.size())
					iEndAngle = anglesDim.size();
				blockDims.iProjAngles = iEndAngle - iAngle;
				assert(blockDims.iProjAngles <= g_MaxAngles);

				//fprintf(stderr, "Major dim %d angle block at %d - %d\n", majordim, iAngle, iEndAngle);
				//fprintf(stderr, "angle indices: %d, %d, ...\n", indicesDim[0], indicesDim[1]);

				ok &= transferConstants(&anglesDim[iAngle], &indicesDim[iAngle], iEndAngle - iAngle, tcbuf, **stream);
				if (!ok)
					break;

				float fStartSlice;
				if (majordim == 0)
					fStartSlice = i - 0.5*dims.iVolX + 0.5;
				else if (majordim == 1)
					fStartSlice = i - 0.5*dims.iVolY + 0.5;
				else if (majordim == 2)
					fStartSlice = i - 0.5*dims.iVolZ + 0.5;

				if (mode != MODE_BACKWARD) {
					ok &= dispatchKernel((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, fStartSlice, dims, blockDims, params, majordim, **stream);
				} else {
					ok &= dispatchKernel_BP((float*)D_projData.ptr, D_projData.pitch/sizeof(float), (float*)D_tmpVolBuffer.ptr, D_tmpVolBuffer.pitch/sizeof(float), fStartSlice, dims, blockDims, params, majordim, **stream);
				}
				if (!ok)
					break;
			}
			if (mode == MODE_BACKWARD) {
				if (majordim == 0) {
					// invert permute + add
					permute_XYZ_to_ZXY_subX(D_tmpVolBuffer, D_volumeData, i, blockDims, dims, true, **stream);
				} else if (majordim == 1) {
					// invert permute + add
					permute_XYZ_to_XZY_subY(D_tmpVolBuffer, D_volumeData, i, blockDims, dims, true, **stream);
				} else if (majordim == 2) {
					// no separate buffer
				}
			}
		}

		// TODO: re-use these if the next major axis happens to have
		// same dimensions. (In particular for cube volumes)
		// (On the other hand, for 3D volumes this re-allocation cost is not that large, relatively)
		if (mode == MODE_FORWARD) {
			logCuda(cudaDestroyTextureObject(D_texObj), "FP3D destroy texture");
			logCuda(cudaFreeArray(D_volArray), "FP3D free array");
		}
		if (mode != MODE_FW_NOINPUT && majordim != 2) {
			logCuda(cudaFree(D_tmpVolBuffer.ptr), "FP3D free buffer");
		}
		if (!ok)
			break;

	}

	ok &= stream.sync("FP3D sync");

	return ok;
}

}
