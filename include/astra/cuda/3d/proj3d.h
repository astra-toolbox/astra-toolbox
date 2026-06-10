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

#ifndef _CUDA_FP3D_H
#define _CUDA_FP3D_H

#include "astra/cuda/3d/dims3d.h"
#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/stream.h"

namespace astraCUDA3d {

// TODO: Clean these up
using astraCUDA::Stream;
using astraCUDA::automatic_sync;

struct SCALE_CUBE {
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1+a2*a2+1.0f) * fOutputScale; }
};

struct SCALE_NONCUBE {
	float fScale1;
	float fScale2;
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1*fScale1+a2*a2*fScale2+1.0f) * fOutputScale; }
};

bool isVolCube(const SProjectorParams3D& params);
SCALE_CUBE getCubeScaling(const SProjectorParams3D& params);
SCALE_NONCUBE getGenericScaling(const SProjectorParams3D& params, int majordim);


// These projector classes behave more like (templated, virtual) namespaces
// than objects currently. Could consider moving a few parameters into the
// classes. (Tunable kernel constants, dimensions, stream, etc)

template<typename TCBUF, typename PROJ>
class RayDrivenProjector3D
{
public:
	virtual ~RayDrivenProjector3D() { }

	enum EMode {
		MODE_FORWARD,
		MODE_FW_NOINPUT,
		MODE_BACKWARD
	};

	bool project(cudaPitchedPtr D_volumeData,
	             cudaPitchedPtr D_projData,
	             EMode mode,
	             const SDimensions3D& dims,
	             const std::vector<PROJ> &angles,
	             const SProjectorParams3D& params,
	             const Stream &stream = Stream(automatic_sync));

protected:
	RayDrivenProjector3D() { }

	virtual bool sortProjections(const SDimensions3D& dims,
	                             const std::vector<PROJ> &angles,
	                             std::vector<PROJ>& anglesX,
	                             std::vector<PROJ>& anglesY,
	                             std::vector<PROJ>& anglesZ,
	                             std::vector<int>& indicesX,
	                             std::vector<int>& indicesY,
	                             std::vector<int>& indicesZ) = 0;

	virtual bool transferConstants(const PROJ* angles, const int* indices,
	                               unsigned int iProjAngles,
	                               TCBUF& buf, cudaStream_t stream) = 0;

	virtual bool dispatchKernel(float *D_projData, unsigned int projPitch,
	                            cudaTextureObject_t tex,
	                            float fStartSlice,
	                            const SDimensions3D& dims,
	                            const SDimensions3D& blockDims,
	                            const SProjectorParams3D& params,
	                            int majordim,
	                            cudaStream_t stream)
	{
		return false;
	}

	virtual bool dispatchKernel_BP(float *D_volData, unsigned int volPitch,
	                               const float *D_projData, unsigned int projPitch,
	                               float fStartSlice,
	                               const SDimensions3D& dims,
	                               const SDimensions3D& blockDims,
	                               const SProjectorParams3D& params,
	                               int majordim,
	                               cudaStream_t stream)
	{
		return false;
	}

};


template<typename TCBUF, typename PROJ>
class VoxelDrivenProjector3D
{
public:
	virtual ~VoxelDrivenProjector3D() { }

	enum EMode {
		MODE_BACKWARD
	};

	bool project(cudaPitchedPtr D_volumeData,
	             cudaPitchedPtr D_projData,
	             EMode mode,
	             bool deviceGeometry,
	             const SDimensions3D& dims,
	             const std::vector<PROJ> &angles,
	             const SProjectorParams3D& params,
	             const Stream &stream = Stream(automatic_sync));

protected:
	VoxelDrivenProjector3D() { }

	virtual bool prepareProjections(cudaPitchedPtr D_tmpProjBuffer,
	                                cudaPitchedPtr D_projData,
	                                const PROJ *D_geom,
	                                unsigned int startAngle,
	                                const SDimensions3D& bufDims,
	                                const SDimensions3D& dims,
	                                const SProjectorParams3D& params,
	                                cudaStream_t stream)
	{
		// Default implementation: permute data to per-angle layers
		return permute_UTV_to_UVT_subT(D_tmpProjBuffer, D_projData, startAngle,
		                               bufDims, dims, false, stream);
	}

	virtual bool transferConstants(const PROJ* angles,
	                               unsigned int iProjAngles,
	                               const SProjectorParams3D& params,
	                               TCBUF& buf, cudaStream_t stream) = 0;

	virtual bool dispatchKernel_BP(float *D_volData, unsigned int volPitch,
	                               cudaTextureObject_t tex,
	                               unsigned int angleCount,
	                               const SDimensions3D& dims,
	                               const SProjectorParams3D& params,
	                               cudaStream_t stream)
	{
		return false;
	}

};

}

#endif
