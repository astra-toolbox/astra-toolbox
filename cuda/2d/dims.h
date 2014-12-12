/*
-----------------------------------------------------------------------
Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
                2014, CWI, Amsterdam

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

#ifndef _CUDA_DIMS_H
#define _CUDA_DIMS_H

namespace astraCUDA {

struct SFanProjection {
        // the source
        float fSrcX, fSrcY;

        // the start of the (linear) detector
        float fDetSX, fDetSY;

        // the length of a single detector pixel
        float fDetUX, fDetUY;
};


struct SDimensions {
	unsigned int iVolWidth;
	unsigned int iVolHeight;
	unsigned int iProjAngles;
	unsigned int iProjDets;
	float fDetScale; // size of detector compared to volume pixels
	unsigned int iRaysPerDet;
	unsigned int iRaysPerPixelDim;
};

}

#endif

