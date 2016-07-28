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

#ifndef _CUDA_SIRT_H
#define _CUDA_SIRT_H

#include "util.h"
#include "algo.h"

namespace astraCUDA {

class _AstraExport SIRT : public ReconAlgo {
public:
	SIRT();
	~SIRT();

	virtual bool init();

	// Do optional long-object compensation. See the comments in sirt.cu.
	// Call this after init(). It can not be used in combination with masks.
	bool doSlabCorrections();

	// Set min/max masks to existing GPU memory buffers
	bool setMinMaxMasks(float* D_minMaskData, float* D_maxMaskData,
	                    unsigned int pitch);

	// Set min/max masks from RAM buffers
	bool uploadMinMaxMasks(const float* minMaskData, const float* maxMaskData,
	                       unsigned int pitch);

	void setRelaxation(float r) { fRelaxation = r; }

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
	float* D_lineWeight;
	unsigned int linePitch;

	float* D_pixelWeight;
	unsigned int pixelPitch;

	// Masks
	bool freeMinMaxMasks;
	float* D_minMaskData;
	unsigned int minMaskPitch;
	float* D_maxMaskData;
	unsigned int maxMaskPitch;

	float fRelaxation;
};

bool doSIRT(float* D_volumeData, unsigned int volumePitch,
            float* D_projData, unsigned int projPitch,
            float* D_maskData, unsigned int maskPitch,
            const SDimensions& dims, const float* angles,
            const float* TOffsets, unsigned int iterations);

}

#endif
