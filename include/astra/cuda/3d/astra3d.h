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

#ifndef _CUDA_ASTRA3D_H
#define _CUDA_ASTRA3D_H

#include "dims3d.h"

namespace astra {



class CFloat32VolumeData3D;
class CFloat32ProjectionData3D;

using astraCUDA3d::Cuda3DProjectionKernel;
using astraCUDA3d::ker3d_default;
using astraCUDA3d::ker3d_matched_bp;
using astraCUDA3d::ker3d_sum_square_weights;
using astraCUDA3d::ker3d_fdk_weighting;
using astraCUDA3d::ker3d_2d_weighting;


_AstraExport bool uploadMultipleProjections(CFloat32ProjectionData3D *proj,
                                            const float *data,
                                            unsigned int y_min,
                                            unsigned int y_max);


}

namespace astraCUDA {
_AstraExport bool setGPUIndex(int index);
}
namespace astraCUDA3d {
using astraCUDA::setGPUIndex;
}


#endif
