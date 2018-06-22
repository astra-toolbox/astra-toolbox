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

#include "astra/cuda/3d/algo3d.h"
#include "astra/cuda/3d/cone_fp.h"
#include "astra/cuda/3d/cone_bp.h"
#include "astra/cuda/3d/par3d_fp.h"
#include "astra/cuda/3d/par3d_bp.h"

#include <cassert>

namespace astraCUDA3d {

ReconAlgo3D::ReconAlgo3D()
{
	coneProjs = 0;
	par3DProjs = 0;
	shouldAbort = false;
}

ReconAlgo3D::~ReconAlgo3D()
{
	reset();
}

void ReconAlgo3D::reset()
{
	delete[] coneProjs;
	coneProjs = 0;
	delete[] par3DProjs;
	par3DProjs = 0;
	shouldAbort = false;
}

bool ReconAlgo3D::setConeGeometry(const SDimensions3D& _dims, const SConeProjection* _angles, const SProjectorParams3D& _params)
{
	dims = _dims;
	params = _params;

	coneProjs = new SConeProjection[dims.iProjAngles];
	par3DProjs = 0;

	memcpy(coneProjs, _angles, sizeof(coneProjs[0]) * dims.iProjAngles);

	return true;
}

bool ReconAlgo3D::setPar3DGeometry(const SDimensions3D& _dims, const SPar3DProjection* _angles, const SProjectorParams3D& _params)
{
	dims = _dims;
	params = _params;

	par3DProjs = new SPar3DProjection[dims.iProjAngles];
	coneProjs = 0;

	memcpy(par3DProjs, _angles, sizeof(par3DProjs[0]) * dims.iProjAngles);

	return true;
}


bool ReconAlgo3D::callFP(cudaPitchedPtr& D_volumeData,
                       cudaPitchedPtr& D_projData,
                       float outputScale)
{
	SProjectorParams3D p = params;
	p.fOutputScale *= outputScale;
	if (coneProjs) {
		return ConeFP(D_volumeData, D_projData, dims, coneProjs, p);
	} else {
		return Par3DFP(D_volumeData, D_projData, dims, par3DProjs, p);
	}
}

bool ReconAlgo3D::callBP(cudaPitchedPtr& D_volumeData,
                       cudaPitchedPtr& D_projData,
                       float outputScale)
{
	SProjectorParams3D p = params;
	p.fOutputScale *= outputScale;
	if (coneProjs) {
		return ConeBP(D_volumeData, D_projData, dims, coneProjs, p);
	} else {
		return Par3DBP(D_volumeData, D_projData, dims, par3DProjs, p);
	}
}



}
