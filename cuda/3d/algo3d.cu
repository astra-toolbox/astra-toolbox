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

#include "astra/cuda/3d/algo3d.h"
#include "astra/cuda/3d/cone_fp.h"
#include "astra/cuda/3d/cone_bp.h"
#include "astra/cuda/3d/cone_cyl.h"
#include "astra/cuda/3d/par3d_fp.h"
#include "astra/cuda/3d/par3d_bp.h"

#include "astra/GeometryUtil3D.h"
#include "astra/Logging.h"

#include <cassert>

namespace astraCUDA3d {

ReconAlgo3D::ReconAlgo3D()
{

}

ReconAlgo3D::~ReconAlgo3D()
{

}

void ReconAlgo3D::reset()
{
	projs.clear();
}

bool ReconAlgo3D::setGeometry(const SDimensions3D& _dims, const astra::Geometry3DParameters& _projs, const SProjectorParams3D& _params)
{
	dims = _dims;
	params = _params;
	projs = _projs;

	return true;
}

bool ReconAlgo3D::callFP(cudaPitchedPtr& D_volumeData,
                       cudaPitchedPtr& D_projData,
                       float outputScale)
{
	SProjectorParams3D p = params;
	p.fOutputScale *= outputScale;
	if (projs.isCone()) {
		return ConeFP(D_volumeData, D_projData, dims, projs.getCone(), p);
	} else if (projs.isParallel()) {
		return Par3DFP(D_volumeData, D_projData, dims, projs.getParallel(), p);
	} else if (projs.isCylCone()) {
		return ConeCylFP(D_volumeData, D_projData, dims, projs.getCylCone(), p);
	} else {
		ASTRA_ERROR("Unsupported geometry type");
		return false;
	}
}

bool ReconAlgo3D::callBP(cudaPitchedPtr& D_volumeData,
                       cudaPitchedPtr& D_projData,
                       float outputScale)
{
	SProjectorParams3D p = params;
	p.fOutputScale *= outputScale;
	if (projs.isCone()) {
		return ConeBP(D_volumeData, D_projData, dims, projs.getCone(), p);
	} else if (projs.isParallel()) {
		return Par3DBP(D_volumeData, D_projData, dims, projs.getParallel(), p);
	} else if (projs.isCylCone()) {
		return ConeCylBP(D_volumeData, D_projData, dims, projs.getCylCone(), p);

	} else {
		ASTRA_ERROR("Unsupported geometry type");
		return false;
	}
}



}
