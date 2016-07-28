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
*/

#ifndef _CUDA_MEM3D_H
#define _CUDA_MEM3D_H

#include <boost/shared_ptr.hpp>

#include "astra3d.h"

namespace astra {
class CVolumeGeometry3D;
class CProjectionGeometry3D;	
}

namespace astraCUDA3d {

// TODO: Make it possible to delete these handles when they're no longer
// necessary inside the FP/BP
//
// TODO: Add functions for querying capacity

struct SMemHandle3D_internal;

struct MemHandle3D {
	boost::shared_ptr<SMemHandle3D_internal> d;
	operator bool() const { return (bool)d; }
};

struct SSubDimensions3D {
	unsigned int nx;
	unsigned int ny;
	unsigned int nz;
	unsigned int pitch;
	unsigned int subnx;
	unsigned int subny;
	unsigned int subnz;
	unsigned int subx;
	unsigned int suby;
	unsigned int subz;
};

/*
// Useful or not?
enum Mem3DCopyMode {
	MODE_SET,
	MODE_ADD
};
*/

enum Mem3DZeroMode {
	INIT_NO,
	INIT_ZERO
};

size_t availableGPUMemory();
int maxBlockDimension();

MemHandle3D allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero);

bool copyToGPUMemory(const float *src, MemHandle3D dst, const SSubDimensions3D &pos);

bool copyFromGPUMemory(float *dst, MemHandle3D src, const SSubDimensions3D &pos);

bool freeGPUMemory(MemHandle3D handle);

bool setGPUIndex(int index);


bool FP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel);

bool BP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iVoxelSuperSampling);



}

#endif
