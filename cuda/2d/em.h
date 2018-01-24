/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#ifndef _CUDA_EM_H
#define _CUDA_EM_H

#include "algo.h"

namespace astraCUDA {

class _AstraExport EM : public ReconAlgo {
public:
	EM();
	virtual ~EM();

	// disable some features
	virtual bool enableSinogramMask() { return false; }
	virtual bool enableVolumeMask() { return false; }
	virtual bool setMinConstraint(float) { return false; }
	virtual bool setMaxConstraint(float) { return false; }

	virtual bool init();

	virtual bool iterate(unsigned int iterations);

	virtual float computeDiffNorm();

protected:
	void reset();
	bool precomputeWeights();

 	// Temporary buffers
	float* D_projData;
	unsigned int projPitch;

	float* D_tmpData;
	unsigned int tmpPitch;

	// Geometry-specific precomputed data
	float* D_pixelWeight;
	unsigned int pixelPitch;
};

_AstraExport bool doEM(float* D_volumeData, unsigned int volumePitch,
          float* D_projData, unsigned int projPitch,
          const SDimensions& dims, const float* angles,
          const float* TOffsets, unsigned int iterations);

}

#endif
