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

#ifndef _CUDA_ASTRA3D_H
#define _CUDA_ASTRA3D_H

#include "dims3d.h"

namespace astra {


// TODO: Switch to a class hierarchy as with the 2D algorithms


enum Cuda3DProjectionKernel {
	ker3d_default = 0,
	ker3d_sum_square_weights
};

class CProjectionGeometry3D;
class CParallelProjectionGeometry3D;
class CParallelVecProjectionGeometry3D;
class CConeProjectionGeometry3D;
class CConeVecProjectionGeometry3D;
class CVolumeGeometry3D;
class AstraSIRT3d_internal;
class CMPIProjector3D;

class _AstraExport AstraSIRT3d {
public:

	AstraSIRT3d();
	~AstraSIRT3d();

	// Set the volume and projection geometry
	bool setGeometry(const CVolumeGeometry3D* pVolGeom,
	                 const CProjectionGeometry3D* pProjGeom,
			 const CMPIProjector3D *pMPIProj = NULL);

	// Enable supersampling.
	//
	// The number of rays used in FP is the square of iDetectorSuperSampling.
	// The number of rays used in BP is the cube of iVoxelSuperSampling.
	bool enableSuperSampling(unsigned int iVoxelSuperSampling,
	                         unsigned int iDetectorSuperSampling);

	// Enable volume/sinogram masks
	//
	// This may optionally be called before init().
	// If it is called, setVolumeMask()/setSinogramMask() must be called between
	// setSinogram() and iterate().
	bool enableVolumeMask();
	bool enableSinogramMask();

	// Set GPU index
	//
	// This should be called before init(). Note that setting the GPU index
	// in a thread which has already used the GPU may not work.
	bool setGPUIndex(int index);

	// Allocate GPU buffers and
	// precompute geometry-specific data.
	//
	// This must be called after calling setReconstructionGeometry() and
	// setProjectionGeometry() or setFanProjectionGeometry().
	bool init();

	// Setup input sinogram for a slice.
	// pfSinogram must be a float array of size XXX
	// NB: iSinogramPitch is measured in floats, not in bytes.
	//
	// This must be called after init(), and before iterate(). It may be
	// called again after iterate()/getReconstruction() to start a new slice.
	//
	// pfSinogram will only be read from during this call.
	bool setSinogram(const float* pfSinogram, unsigned int iSinogramPitch);

	// Setup volume mask for a slice.
	// pfMask must be a float array of size XXX
	// NB: iMaskPitch is measured in floats, not in bytes.
	//
	// It may only contain the exact values 0.0f and 1.0f. Only volume pixels
	// for which pfMask[z] is 1.0f are processed.
	bool setVolumeMask(const float* pfMask, unsigned int iMaskPitch);

	// Setup sinogram mask for a slice.
	// pfMask must be a float array of size XXX
	// NB: iMaskPitch is measured in floats, not in bytes.
	//
	// It may only contain the exact values 0.0f and 1.0f. Only sinogram pixels
	// for which pfMask[z] is 1.0f are processed.
	bool setSinogramMask(const float* pfMask, unsigned int iMaskPitch);

	// Set the starting reconstruction for SIRT.	
	// pfReconstruction must be a float array of size XXX
	// NB: iReconstructionPitch is measured in floats, not in bytes.
	//
	// This may be called between setSinogram() and iterate().
	// If this function is not called before iterate(), SIRT will start
	// from a zero reconstruction.
	//
	// pfReconstruction will only be read from during this call.
	bool setStartReconstruction(const float* pfReconstruction,
	                            unsigned int iReconstructionPitch);

	// Enable min/max constraint.
	//
	// These may optionally be called between init() and iterate()
	bool setMinConstraint(float fMin);
	bool setMaxConstraint(float fMax);

	// Perform a number of (additive) SIRT iterations.
	// This must be called after setSinogram().
	//
	// If called multiple times, without calls to setSinogram() or
	// setStartReconstruction() in between, iterate() will continue from
	// the result of the previous call.
	// Calls to getReconstruction() are allowed between calls to iterate() and
	// do not change the state.
	bool iterate(unsigned int iIterations);

	// Get the reconstructed slice.
	// pfReconstruction must be a float array of size XXX
	// NB: iReconstructionPitch is measured in floats, not in bytes.
	//
	// This may be called after iterate().
	bool getReconstruction(float* pfReconstruction,
	                       unsigned int iReconstructionPitch) const;

	// Compute the norm of the difference of the FP of the current
	// reconstruction and the sinogram. (This performs one FP.)
	// It can be called after iterate().
	float computeDiffNorm();

	// Signal the algorithm that it should abort after the current iteration.
	// This is intended to be called from another thread.
	void signalAbort();

protected:
	AstraSIRT3d_internal *pData;
};


class AstraCGLS3d_internal;


class _AstraExport AstraCGLS3d {
public:

	AstraCGLS3d();
	~AstraCGLS3d();

	// Set the volume and projection geometry
	bool setGeometry(const CVolumeGeometry3D* pVolGeom,
	                 const CProjectionGeometry3D* pProjGeom,
			 const CMPIProjector3D *pMPIProj = NULL);

	// Enable supersampling.
	//
	// The number of rays used in FP is the square of iDetectorSuperSampling.
	// The number of rays used in BP is the cube of iVoxelSuperSampling.
	bool enableSuperSampling(unsigned int iVoxelSuperSampling,
	                         unsigned int iDetectorSuperSampling);

	// Enable volume/sinogram masks
	//
	// This may optionally be called before init().
	// If it is called, setVolumeMask()/setSinogramMask() must be called between
	// setSinogram() and iterate().
	bool enableVolumeMask();
	//bool enableSinogramMask();

	// Set GPU index
	//
	// This should be called before init(). Note that setting the GPU index
	// in a thread which has already used the GPU may not work.
	bool setGPUIndex(int index);

	// Allocate GPU buffers and
	// precompute geometry-specific data.
	//
	// This must be called after calling setReconstructionGeometry() and
	// setProjectionGeometry() or setFanProjectionGeometry().
	bool init();

	// Setup input sinogram for a slice.
	// pfSinogram must be a float array of size XXX
	// NB: iSinogramPitch is measured in floats, not in bytes.
	//
	// This must be called after init(), and before iterate(). It may be
	// called again after iterate()/getReconstruction() to start a new slice.
	//
	// pfSinogram will only be read from during this call.
	bool setSinogram(const float* pfSinogram, unsigned int iSinogramPitch);

	// Setup volume mask for a slice.
	// pfMask must be a float array of size XXX
	// NB: iMaskPitch is measured in floats, not in bytes.
	//
	// It may only contain the exact values 0.0f and 1.0f. Only volume pixels
	// for which pfMask[z] is 1.0f are processed.
	bool setVolumeMask(const float* pfMask, unsigned int iMaskPitch);

	// Setup sinogram mask for a slice.
	// pfMask must be a float array of size XXX
	// NB: iMaskPitch is measured in floats, not in bytes.
	//
	// It may only contain the exact values 0.0f and 1.0f. Only sinogram pixels
	// for which pfMask[z] is 1.0f are processed.
	//bool setSinogramMask(const float* pfMask, unsigned int iMaskPitch);

	// Set the starting reconstruction for SIRT.	
	// pfReconstruction must be a float array of size XXX
	// NB: iReconstructionPitch is measured in floats, not in bytes.
	//
	// This may be called between setSinogram() and iterate().
	// If this function is not called before iterate(), SIRT will start
	// from a zero reconstruction.
	//
	// pfReconstruction will only be read from during this call.
	bool setStartReconstruction(const float* pfReconstruction,
	                            unsigned int iReconstructionPitch);

	// Enable min/max constraint.
	//
	// These may optionally be called between init() and iterate()
	//bool setMinConstraint(float fMin);
	//bool setMaxConstraint(float fMax);

	// Perform a number of (additive) SIRT iterations.
	// This must be called after setSinogram().
	//
	// If called multiple times, without calls to setSinogram() or
	// setStartReconstruction() in between, iterate() will continue from
	// the result of the previous call.
	// Calls to getReconstruction() are allowed between calls to iterate() and
	// do not change the state.
	bool iterate(unsigned int iIterations);

	// Get the reconstructed slice.
	// pfReconstruction must be a float array of size XXX
	// NB: iReconstructionPitch is measured in floats, not in bytes.
	//
	// This may be called after iterate().
	bool getReconstruction(float* pfReconstruction,
	                       unsigned int iReconstructionPitch) const;

	// Compute the norm of the difference of the FP of the current
	// reconstruction and the sinogram. (This performs one FP.)
	// It can be called after iterate().
	float computeDiffNorm();

	// Signal the algorithm that it should abort after the current iteration.
	// This is intended to be called from another thread.
	void signalAbort();

protected:
	AstraCGLS3d_internal *pData;
};

bool convertAstraGeometry_dims(const CVolumeGeometry3D* pVolGeom,
                               const CProjectionGeometry3D* pProjGeom,
                               astraCUDA3d::SDimensions3D& dims);

bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CProjectionGeometry3D* pProjGeom,
                          SPar3DProjection*& pParProjs,
                          SConeProjection*& pConeProjs,
                          float& fOutputScale);

_AstraExport bool astraCudaFP(const float* pfVolume, float* pfProjections,
                      const CVolumeGeometry3D* pVolGeom,
                      const CProjectionGeometry3D* pProjGeom,
                      int iGPUIndex, int iDetectorSuperSampling,
                      Cuda3DProjectionKernel projKernel,
		      const CMPIProjector3D *pMPIPrj = NULL);


_AstraExport bool astraCudaBP(float* pfVolume, const float* pfProjections,
                      const CVolumeGeometry3D* pVolGeom,
                      const CProjectionGeometry3D* pProjGeom,
                      int iGPUIndex, int iVoxelSuperSampling,
		      const CMPIProjector3D *pMPIPrj = NULL);

_AstraExport bool astraCudaBP_SIRTWeighted(float* pfVolume, const float* pfProjections,
                      const CVolumeGeometry3D* pVolGeom,
                      const CProjectionGeometry3D* pProjGeom,
                      int iGPUIndex, int iVoxelSuperSampling);

_AstraExport bool astraCudaFDK(float* pfVolume, const float* pfProjections,
                  const CVolumeGeometry3D* pVolGeom,
                  const CConeProjectionGeometry3D* pProjGeom,
                  bool bShortScan,
                  int iGPUIndex, int iVoxelSuperSampling);


}


#endif
