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

#ifndef _CUDA_ALGO_H
#define _CUDA_ALGO_H

#include "util.h"

namespace astraCUDA {

class _AstraExport ReconAlgo {
public:
	ReconAlgo();
	virtual ~ReconAlgo();

	bool setGPUIndex(int iGPUIndex);

	bool setGeometry(const SDimensions& dims, const float* angles);
	bool setFanGeometry(const SDimensions& dims, const SFanProjection* projs);

	// setTOffsets should (optionally) be called after setGeometry
	bool setTOffsets(const float* TOffsets);

	void signalAbort() { shouldAbort = true; }

	virtual bool enableVolumeMask();
	virtual bool enableSinogramMask();

	// init should be called after setting all geometry
	virtual bool init() = 0;

	// setVolumeMask should be called after init and before iterate,
	// but only if enableVolumeMask was called before init.
	// It may be called again after iterate.
	bool setVolumeMask(float* D_maskData, unsigned int maskPitch);

	// setSinogramMask should be called after init and before iterate,
	// but only if enableSinogramMask was called before init.
	// It may be called again after iterate.
	bool setSinogramMask(float* D_smaskData, unsigned int smaskPitch);


	// setBuffers should be called after init and before iterate.
	// It may be called again after iterate.
	virtual bool setBuffers(float* D_volumeData, unsigned int volumePitch,
	                        float* D_projData, unsigned int projPitch);


	// instead of calling setBuffers, you can also call allocateBuffers
	// to let ReconAlgo manage its own GPU memory
	virtual bool allocateBuffers();

	// copy data to GPU. This must be called after allocateBuffers.
	// pfSinogram, pfReconstruction, pfVolMask, pfSinoMask are the
	// sinogram, reconstruction, volume mask and sinogram mask in system RAM,
	// respectively. The corresponding pitch variables give the pitches
	// of these buffers, measured in floats.
	// The sinogram is multiplied by fSinogramScale after uploading it.
	virtual bool copyDataToGPU(const float* pfSinogram, unsigned int iSinogramPitch, float fSinogramScale,
	                           const float* pfReconstruction, unsigned int iReconstructionPitch,
	                           const float* pfVolMask, unsigned int iVolMaskPitch,
	                           const float* pfSinoMask, unsigned int iSinoMaskPitch);



	// set Min/Max constraints. They may be called at any time, and will affect
	// any iterate() calls afterwards.
	virtual bool setMinConstraint(float fMin);
	virtual bool setMaxConstraint(float fMax);


	// iterate should be called after init and setBuffers.
	// It may be called multiple times.
	virtual bool iterate(unsigned int iterations) = 0;

	// Compute the norm of the difference of the FP of the current
	// reconstruction and the sinogram. (This performs one FP.)
	// It can be called after iterate.
	virtual float computeDiffNorm() = 0;
	// TODO: computeDiffNorm shouldn't be virtual, but for it to be
	// implemented in ReconAlgo, it needs a way to get a suitable
	// temporary sinogram buffer.

	bool getReconstruction(float* pfReconstruction,
                           unsigned int iReconstructionPitch) const;



protected:
	void reset();

	bool callFP(float* D_volumeData, unsigned int volumePitch,
	            float* D_projData, unsigned int projPitch,
	            float outputScale);
	bool callBP(float* D_volumeData, unsigned int volumePitch,
	            float* D_projData, unsigned int projPitch,
	            float outputScale);


	SDimensions dims;
	float* angles;
	float* TOffsets;
	SFanProjection* fanProjs;

	volatile bool shouldAbort;

	bool freeGPUMemory;

	// Input/output
	float* D_sinoData;
	unsigned int sinoPitch;

	float* D_volumeData;
	unsigned int volumePitch;

	// Masks
	bool useVolumeMask;
	bool useSinogramMask;

	float* D_maskData;
	unsigned int maskPitch;
	float* D_smaskData;
	unsigned int smaskPitch;

	// Min/max
	bool useMinConstraint;
	bool useMaxConstraint;
	float fMinConstraint;
	float fMaxConstraint;


};


}

#endif

