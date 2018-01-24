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

#ifndef MEXCOPYDATAHELPFUNCTIONS_H_
#define MEXCOPYDATAHELPFUNCTIONS_H_

#include <mex.h>

#include "astra/Globals.h"

#include <vector>

void copyMexToCFloat32Array(const mxArray * const, astra::float32 * const,
		const size_t &);
void copyCFloat32ArrayToMex(const astra::float32 * const, mxArray * const);

template<mxClassID MType, class AType>
mxArray * createEquivMexArray(const AType * const pDataObj)
{
	mwSize dims[3];
	dims[0] = pDataObj->getWidth();
	dims[1] = pDataObj->getHeight();
	dims[2] = pDataObj->getDepth();

	return mxCreateNumericArray(3, dims, MType, mxREAL);
}

#endif /* MEXCOPYDATAHELPFUNCTIONS_H_ */
