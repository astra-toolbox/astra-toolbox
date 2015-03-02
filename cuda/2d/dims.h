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

#ifndef _CUDA_DIMS_H
#define _CUDA_DIMS_H

#include "astra/GeometryUtil2D.h"


namespace astraCUDA {

using astra::SFanProjection;


struct SDimensions {
	// Width, height of reconstruction volume
	unsigned int iVolWidth;
	unsigned int iVolHeight;

	// Number of projection angles
	unsigned int iProjAngles;

	// Number of detector pixels
	unsigned int iProjDets;

	// size of detector compared to volume pixels
	float fDetScale;

	// in FP, number of rays to trace per detector pixel.
	// This should usually be set to 1.
	// If fDetScale > 1, this should be set to an integer of roughly
	// the same size as fDetScale.
	unsigned int iRaysPerDet;

	// in BP, square root of number of rays to trace per volume pixel
	// This should usually be set to 1.
	// If fDetScale < 1, this should be set to an integer of roughly
	// the same size as 1 / fDetScale.
	unsigned int iRaysPerPixelDim;
};

}

#endif

