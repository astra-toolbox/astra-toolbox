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

#ifndef _CUDA_CONE_DIMS_H
#define _CUDA_CONE_DIMS_H

#include "astra/GeometryUtil3D.h"


namespace astraCUDA3d {

using astra::SConeProjection;
using astra::SPar3DProjection;


enum Cuda3DProjectionKernel {
	ker3d_default = 0,
	ker3d_sum_square_weights
};


struct SDimensions3D {
	unsigned int iVolX;
	unsigned int iVolY;
	unsigned int iVolZ;
	unsigned int iProjAngles;
	unsigned int iProjU; // number of detectors in the U direction
	unsigned int iProjV; // number of detectors in the V direction
};

struct SProjectorParams3D {
	SProjectorParams3D() :
	    iRaysPerDetDim(1), iRaysPerVoxelDim(1),
	    fOutputScale(1.0f),
	    fVolScaleX(1.0f), fVolScaleY(1.0f), fVolScaleZ(1.0f),
	    ker(ker3d_default),
	    bFDKWeighting(false)
	{ }

	unsigned int iRaysPerDetDim;
	unsigned int iRaysPerVoxelDim;
	float fOutputScale;
	float fVolScaleX;
	float fVolScaleY;
	float fVolScaleZ;
	Cuda3DProjectionKernel ker;
	bool bFDKWeighting;
};

}

#endif

