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

#ifndef _CUDA_DIMS3D_H
#define _CUDA_DIMS3D_H

#include "astra/GeometryUtil3D.h"


namespace astraCUDA3d {

using astra::SConeProjection;
using astra::SCylConeProjection;
using astra::SPar3DProjection;


using astra::SDimensions3D;
using astra::SVolScale3D;


enum Cuda3DProjectionKernel {
	ker3d_default = 0,
	ker3d_sum_square_weights,
	ker3d_fdk_weighting,
	ker3d_2d_weighting,
	ker3d_matched_bp
};

struct SProjectorParams3D {
	SProjectorParams3D() :
        volScale(), 
	    iRaysPerDetDim(1), iRaysPerVoxelDim(1),
	    fOutputScale(1.0f),
	    projKernel(ker3d_default)
	{ }

	SVolScale3D volScale;
	unsigned int iRaysPerDetDim;
	unsigned int iRaysPerVoxelDim;
	float fOutputScale;
	Cuda3DProjectionKernel projKernel;
};

}

#endif

