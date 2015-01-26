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

#ifndef _CUDA_ARITH2_H
#define _CUDA_ARITH2_H

#include "util.h"

namespace astraCUDA {

	void roiSelect(float* out, float radius, unsigned int width, unsigned int height);
	void dartMask(float* out, const float* in, unsigned int conn, unsigned int radius, unsigned int threshold, unsigned int width, unsigned int height);
	void dartSmoothing(float* out, const float* in, float b, unsigned int radius, unsigned int width, unsigned int height);

	_AstraExport bool setGPUIndex(int index);

}

#endif
