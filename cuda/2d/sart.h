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

#ifndef _CUDA_SART_H
#define _CUDA_SART_H

#include "util.h"
#include "algo.h"

namespace astraCUDA {

class _AstraExport SART : public ReconAlgo {
public:
	SART();
	~SART();

	// disable some features
	virtual bool enableSinogramMask() { return false; }

	virtual bool init();

	virtual bool setProjectionOrder(int* projectionOrder, int projectionCount);

	virtual bool iterate(unsigned int iterations);

	virtual float computeDiffNorm();

	void setRelaxation(float r) { fRelaxation = r; }

protected:
	void reset();
	bool precomputeWeights();

	bool callFP_SART(float* D_volumeData, unsigned int volumePitch,
	                 float* D_projData, unsigned int projPitch,
	                 unsigned int angle, float outputScale);
	bool callBP_SART(float* D_volumeData, unsigned int volumePitch,
	                 float* D_projData, unsigned int projPitch,
	                 unsigned int angle, float outputScale);


	// projection angle variables
	bool customOrder;
	int* projectionOrder;
	int projectionCount;
	int iteration;

 	// Temporary buffers
	float* D_projData;
	unsigned int projPitch;

	float* D_tmpData; // Only used when there's a volume mask
	unsigned int tmpPitch;

	// Geometry-specific precomputed data
	float* D_lineWeight;
	unsigned int linePitch;

	float fRelaxation;
};

}

#endif
