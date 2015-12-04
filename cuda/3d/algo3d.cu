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

#include <cassert>

#include "algo3d.h"
#include "cone_fp.h"
#include "cone_bp.h"
#include "par3d_fp.h"
#include "par3d_bp.h"

namespace astraCUDA3d {

ReconAlgo3D::ReconAlgo3D()
{
	coneProjs = 0;
	par3DProjs = 0;
	shouldAbort = false;
	fOutputScale = 1.0f;
        mpiPrj = NULL;
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

bool ReconAlgo3D::setConeGeometry(const SDimensions3D& _dims, const SConeProjection* _angles, float _outputScale)
{
	dims = _dims;
	fOutputScale = _outputScale;

	coneProjs = new SConeProjection[dims.iProjAngles];
	par3DProjs = 0;

	memcpy(coneProjs, _angles, sizeof(coneProjs[0]) * dims.iProjAngles);

	return true;
}

bool ReconAlgo3D::setPar3DGeometry(const SDimensions3D& _dims, const SPar3DProjection* _angles, float _outputScale)
{
	dims = _dims;
	fOutputScale = _outputScale;

	par3DProjs = new SPar3DProjection[dims.iProjAngles];
	coneProjs = 0;

	memcpy(par3DProjs, _angles, sizeof(par3DProjs[0]) * dims.iProjAngles);

	return true;
}


bool ReconAlgo3D::callFP(cudaPitchedPtr& D_volumeData,
                       cudaPitchedPtr& D_projData,
                       float outputScale)
{
	if (coneProjs) {
		return ConeFP(D_volumeData, D_projData, dims, coneProjs, outputScale * this->fOutputScale, mpiPrj);
	} else {
		return Par3DFP(D_volumeData, D_projData, dims, par3DProjs, outputScale * this->fOutputScale, mpiPrj);
	}
}

bool ReconAlgo3D::callBP(cudaPitchedPtr& D_volumeData,
                       cudaPitchedPtr& D_projData,
                       float outputScale)
{
	if (coneProjs) {
		return ConeBP(D_volumeData, D_projData, dims, coneProjs, outputScale * this->fOutputScale, mpiPrj);
	} else {
		return Par3DBP(D_volumeData, D_projData, dims, par3DProjs, outputScale * this->fOutputScale, mpiPrj);
	}
}



}
