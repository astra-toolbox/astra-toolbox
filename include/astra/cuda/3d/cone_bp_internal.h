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

#ifndef _CUDA_CONE_BP_INTERNAL_H
#define _CUDA_CONE_BP_INTERNAL_H

#include "astra/cuda/3d/proj3d.h"

namespace astraCUDA3d {

struct DevConeParams {
	float4 fNumU;
	float4 fNumV;
	float4 fDen;
};

class ConeBP_t : public VoxelDrivenProjector3D<TransferConstantsBuffer_t<DevConeParams>, SConeProjection> {
public:
	virtual bool prepareProjections(cudaPitchedPtr D_tmpProjBuffer,
	                                cudaPitchedPtr D_projData,
	                                const SConeProjection *D_geom,
	                                unsigned int startAngle,
	                                const SDimensions3D& bufDims,
	                                const SDimensions3D& dims,
	                                const SProjectorParams3D& params,
	                                cudaStream_t stream) override;

	virtual bool transferConstants(const SConeProjection* angles,
	                               unsigned int iProjAngles,
	                               const SProjectorParams3D& params,
	                               TransferConstantsBuffer_t<DevConeParams>& buf,
	                               cudaStream_t stream) override;

	virtual bool dispatchKernel_BP(float *D_volData, unsigned int volPitch,
	                               cudaTextureObject_t tex,
	                               unsigned int angleCount,
	                               const SDimensions3D& dims,
	                               const SProjectorParams3D& params,
	                               cudaStream_t stream) override;
};



}

#endif
