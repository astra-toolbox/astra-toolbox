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

// Combined "outer loop" for calling voxel-driven kernels. It processes a
// small block of angles at a time, copies those into a temp buffer,
// permuting axes, and binds a layered texture to the result.

template<typename TCBUF, typename PROJ>
bool VoxelDrivenProjector3D<TCBUF,PROJ>::project(cudaPitchedPtr D_volumeData,
                                                 cudaPitchedPtr D_projData,
                                                 VoxelDrivenProjector3D::EMode mode,
                                                 bool deviceGeometry,
                                                 const SDimensions3D& dims,
                                                 const std::vector<PROJ> &angles,
                                                 const SProjectorParams3D& params_,
                                                 const Stream &stream)
{
	if (!stream.isValid())
		return false;
	SProjectorParams3D params = params_;

	// Allocate buffers, swapped to U,V,Angle axis order

	SDimensions3D bufdims = dims;
	// Allocate smaller number of angles when dims.iProjAngles < g_MaxAngles
	bufdims.iProjAngles = std::min(g_MaxAngles, dims.iProjAngles);

	cudaArray* D_projArray = allocateProjectionArray_LayeredAngle(bufdims);
	if (!D_projArray)
		return false;

	cudaPitchedPtr D_tmpProjBuffer = allocateProjectionData_UVAngle(bufdims);
	if (!D_tmpProjBuffer.ptr) {
		logCuda(cudaFreeArray(D_projArray), "BP3D free array");
		return false;
	}

	PROJ* D_geom = nullptr;
	if (deviceGeometry) {
		if (!checkCuda(cudaMalloc((void**)&D_geom, sizeof(PROJ) * dims.iProjAngles), "BP3D geom alloc")) {
			logCuda(cudaFreeArray(D_projArray), "BP3D free array");
			logCuda(cudaFree(D_tmpProjBuffer.ptr), "BP3D free buffer");
			return false;
		}

		// TODO: async?

		if (!checkCuda(cudaMemcpy(D_geom, &angles[0], sizeof(PROJ) * dims.iProjAngles, cudaMemcpyHostToDevice), "BP3D geom memcpy")) {
			logCuda(cudaFreeArray(D_projArray), "BP3D free array");
			logCuda(cudaFree(D_tmpProjBuffer.ptr), "BP3D free buffer");
			if (deviceGeometry)
				logCuda(cudaFree(D_geom), "BP3D free geom buffer");
			return false;
		}
	}

	TCBUF tcbuf(bufdims.iProjAngles);

	cudaTextureObject_t D_texObj;
	if (!createTextureObject3D(D_projArray, D_texObj)) {
		logCuda(cudaFreeArray(D_projArray), "BP3D free array");
		logCuda(cudaFree(D_tmpProjBuffer.ptr), "BP3D free buffer");
		if (deviceGeometry)
			logCuda(cudaFree(D_geom), "BP3D free geom buffer");
		return false;
	}

	// HACK: output scaling for FDK:
	// TODO: Might be cleaner to put this scaling elsewhere
	if (params.projKernel == ker3d_fdk_weighting) {
		// NB: assuming cube voxels here
		params.fOutputScale /= params.volScale.fX;
	} else {
		params.fOutputScale *= (params.volScale.fX * params.volScale.fY * params.volScale.fZ);
	}



	bool ok = true;

	for (unsigned int th = 0; th < dims.iProjAngles; th += g_MaxAngles) {
		unsigned int angleCount = g_MaxAngles;
		if (th + angleCount > dims.iProjAngles)
			angleCount = dims.iProjAngles - th;
		assert(angleCount <= bufdims.iProjAngles);

		// Re-shuffle current set of angles into D_tmpProjBuffer
		ok &= prepareProjections(D_tmpProjBuffer, D_projData, D_geom, th, bufdims, dims, params, **stream);
		if (!ok)
			break;

		ok &= transferProjectionsToArray_LayeredAngle(D_tmpProjBuffer, D_projArray, bufdims, **stream);
		if (!ok)
			break;

		ok = transferConstants(&angles[th], angleCount, params, tcbuf, **stream);
		if (!ok)
			break;

		// Without an explicit sync here, the texture (cache?) doesn't seem
		// to be refreshed for the new batch of angles
		ok &= stream.sync("BP3D sync");
		if (!ok)
			break;

		ok &= dispatchKernel_BP((float*)D_volumeData.ptr, D_volumeData.pitch/sizeof(float), D_texObj, angleCount, dims, params, **stream);
		if (!ok)
			break;
	}

	// TODO: Consider replacing this with stream.syncIfAuto()
	// (after ensuring memory management is synced properly)
	ok &= stream.sync("BP3D sync");

	logCuda(cudaDestroyTextureObject(D_texObj), "BP3D destroy texture");
	logCuda(cudaFreeArray(D_projArray), "BP3D free array");
	logCuda(cudaFree(D_tmpProjBuffer.ptr), "BP3D free buffer");
	if (deviceGeometry)
		logCuda(cudaFree(D_geom), "BP3D free geom buffer");

	return ok;
}


}
