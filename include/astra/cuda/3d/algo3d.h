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

#ifndef _CUDA_ALGO3D_H
#define _CUDA_ALGO3D_H

#include "dims3d.h"
#include "util3d.h"

namespace astra {
class GeometryParameters3D;
}

namespace astraCUDA3d {

class _AstraExport ReconAlgo3D {
public:
	ReconAlgo3D();
	~ReconAlgo3D();

	bool setGeometry(const SDimensions3D& dims, const astra::Geometry3DParameters &projs, const SProjectorParams3D& params);

protected:
	void reset();

	bool callFP(cudaPitchedPtr& D_volumeData, 
	            cudaPitchedPtr& D_projData, 
	            float outputScale);
	bool callBP(cudaPitchedPtr& D_volumeData, 
	            cudaPitchedPtr& D_projData,
	            float outputScale);

	SDimensions3D dims;
	SProjectorParams3D params;
	astra::Geometry3DParameters projs;

	float fOutputScale;
};


}

#endif

