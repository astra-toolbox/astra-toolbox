/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
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

#ifndef MEXDATAMANAGERHELPFUNCTIONS_H_
#define MEXDATAMANAGERHELPFUNCTIONS_H_

#include <mex.h>

#include "astra/Globals.h"
#include "astra/AstraObjectManager.h"
#include "astra/Float32Data3DMemory.h"
#include "astra/ProjectionGeometry3D.h"
#include "astra/VolumeGeometry3D.h"

#include "mexCopyDataHelpFunctions.h"

bool checkID(const astra::int32 &, astra::CFloat32Data3DMemory *&);

bool checkDataType(const mxArray * const);
bool checkStructs(const mxArray * const);

bool checkDataSize(const mxArray * const, const astra::CProjectionGeometry3D * const);
bool checkDataSize(const mxArray * const, const astra::CVolumeGeometry3D * const);
bool checkDataSize(const mxArray * const, const astra::CProjectionGeometry3D * const,
		const mwIndex & zOffset);
bool checkDataSize(const mxArray * const, const astra::CVolumeGeometry3D * const,
		const mwIndex & zOffset);

void getDataPointers(const std::vector<astra::CFloat32Data3DMemory *> &,
		std::vector<astra::float32 *> &);
void getDataSizes(const std::vector<astra::CFloat32Data3DMemory *> &,
		std::vector<size_t> &);

astra::CFloat32Data3DMemory * allocateDataObject(const std::string & sDataType,
		const mxArray * const geometry, const mxArray * const data,
		const mxArray * const unshare = NULL, const mxArray * const zOffset = NULL);

//-----------------------------------------------------------------------------------------
template<mxClassID datatype>
void generic_astra_mex_data3d_get(int nlhs, mxArray* plhs[], int nrhs,
		const mxArray* prhs[])
{
	// step1: input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	// step2: get data object/s
	astra::CFloat32Data3DMemory* pDataObject = NULL;
	if (!checkID(mxGetScalar(prhs[1]), pDataObject)) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		plhs[0] = createEquivMexArray<datatype>(pDataObject);
		copyCFloat32ArrayToMex(pDataObject->getData(), plhs[0]);
	}
}

#endif /* MEXDATAMANAGERHELPFUNCTIONS_H_ */
