/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#ifndef _CUDA_CGLS3D_H
#define _CUDA_CGLS3D_H

#include "util3d.h"
#include "algo3d.h"

namespace astraCUDA3d {

class _AstraExport CGLS : public ReconAlgo3D {
public:
	CGLS();
	~CGLS();

//	bool setConeGeometry(const SDimensions3D& dims, const SConeProjection* projs);


	bool enableVolumeMask();
	bool enableSinogramMask();

	// init should be called after setting all geometry
	bool init();

	// setVolumeMask should be called after init and before iterate,
	// but only if enableVolumeMask was called before init.
	// It may be called again after iterate.
	bool setVolumeMask(cudaPitchedPtr& D_maskData);

	// setSinogramMask should be called after init and before iterate,
	// but only if enableSinogramMask was called before init.
	// It may be called again after iterate.
	bool setSinogramMask(cudaPitchedPtr& D_smaskData);


	// setBuffers should be called after init and before iterate.
	// It may be called again after iterate.
	bool setBuffers(cudaPitchedPtr& D_volumeData,
	                cudaPitchedPtr& D_projData);


	// set Min/Max constraints. They may be called at any time, and will affect
	// any iterate() calls afterwards.
	bool setMinConstraint(float fMin) { return false; }
	bool setMaxConstraint(float fMax) { return false; }

	// iterate should be called after init and setBuffers.
	// It may be called multiple times.
	bool iterate(unsigned int iterations);

	// Compute the norm of the difference of the FP of the current reconstruction
	// and the sinogram. (This performs one FP.)
	// It can be called after iterate.
	float computeDiffNorm();

protected:
	void reset();

	bool useVolumeMask;
	bool useSinogramMask;

	cudaPitchedPtr D_maskData;
	cudaPitchedPtr D_smaskData;

	// Input/output
	cudaPitchedPtr D_sinoData;
	cudaPitchedPtr D_volumeData;

 	// Temporary buffers
	cudaPitchedPtr D_r;
	cudaPitchedPtr D_w;
	cudaPitchedPtr D_z;
	cudaPitchedPtr D_p;

	float gamma;

	bool sliceInitialized;
};

_AstraExport bool doCGLS(cudaPitchedPtr D_volumeData, unsigned int volumePitch,
            cudaPitchedPtr D_projData, unsigned int projPitch,
            cudaPitchedPtr D_maskData, unsigned int maskPitch,
            const SDimensions3D& dims, const SConeProjection* projs,
            unsigned int iterations);

}

#endif
