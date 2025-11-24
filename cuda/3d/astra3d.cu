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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/arith3d.h"
#include "astra/cuda/3d/astra3d.h"
#include "astra/cuda/3d/mem3d.h"

#include "astra/Data3D.h"
#include "astra/Logging.h"

#include "astra/GeometryUtil3D.h"

#include <iostream>
#include <cstdio>
#include <cassert>

using namespace astraCUDA3d;

namespace astra {


_AstraExport bool uploadMultipleProjections(CFloat32ProjectionData3D *proj,
                                         const float *data,
                                         unsigned int y_min, unsigned int y_max)
{
	astraCUDA3d::SDimensions3D dims1;
	dims1.iProjU = proj->getDetectorColCount();
	dims1.iProjV = proj->getDetectorRowCount();
	dims1.iProjAngles = y_max - y_min + 1;

	cudaPitchedPtr D_proj = allocateProjectionData(dims1);
	bool ok = copyProjectionsToDevice(data, D_proj, dims1);
	if (!ok) {
		ASTRA_ERROR("Failed to upload projection to GPU");
		return false;
	}

	CDataStorage *storage = astraCUDA3d::wrapHandle(
			(float *)D_proj.ptr,
			dims1.iProjU, dims1.iProjAngles, dims1.iProjV,
			D_proj.pitch / sizeof(float));
	CData3D *inputData = new CData3D(dims1.iProjU, dims1.iProjAngles, dims1.iProjV, storage);


	astraCUDA3d::SSubDimensions3D subdims;
	subdims.nx = dims1.iProjU;
	subdims.ny = proj->getAngleCount();
	subdims.nz = dims1.iProjV;
	subdims.pitch = D_proj.pitch / sizeof(float); // FIXME: Pitch for wrong obj!
	subdims.subnx = dims1.iProjU;
	subdims.subny = dims1.iProjAngles;
	subdims.subnz = dims1.iProjV;
	subdims.subx = 0;
	subdims.suby = y_min;
	subdims.subz = 0;

	ok = astraCUDA3d::copyIntoArray(proj, inputData, subdims);
	if (!ok) {
		ASTRA_ERROR("Failed to copy projection into 3d data");
		return false;
	}

	cudaFree(D_proj.ptr);
	return true;
}


}
