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

#include "astra/cuda/3d/proj3d.h"
#include "astra/cuda/3d/util3d.h"

namespace astraCUDA3d {

bool isVolCube(const SProjectorParams3D& params)
{
	bool cube = true;
	if (abs(params.volScale.fX / params.volScale.fY - 1.0) > 0.00001)
		cube = false;
	if (abs(params.volScale.fX / params.volScale.fZ - 1.0) > 0.00001)
		cube = false;
	return cube;
}

SCALE_CUBE getCubeScaling(const SProjectorParams3D& params)
{
	SCALE_CUBE scube;
	scube.fOutputScale = params.fOutputScale * params.volScale.fX;
	return scube;
}

SCALE_NONCUBE getGenericScaling(const SProjectorParams3D& params, int majordim)
{
	SCALE_NONCUBE snoncube;
	float fS1, fS2;
	if (majordim == 0) {
		fS1 = params.volScale.fY / params.volScale.fX;
		fS2 = params.volScale.fZ / params.volScale.fX;
		snoncube.fOutputScale = params.fOutputScale * params.volScale.fX;
	} else if (majordim == 1) {
		fS1 = params.volScale.fX / params.volScale.fY;
		fS2 = params.volScale.fZ / params.volScale.fY;
		snoncube.fOutputScale = params.fOutputScale * params.volScale.fY;
	} else if (majordim == 2) {
		fS1 = params.volScale.fX / params.volScale.fZ;
		fS2 = params.volScale.fY / params.volScale.fZ;
		snoncube.fOutputScale = params.fOutputScale * params.volScale.fZ;
	}
	snoncube.fScale1 = fS1 * fS1;
	snoncube.fScale2 = fS2 * fS2;

	return snoncube;
}

}
