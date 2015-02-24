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

/** \file astra_mex_data3d_c.cpp
 *
 *  \brief Creates, manages and manipulates 3D volume and projection data objects.
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexCopyDataHelpFunctions.h"
#include "mexDataManagerHelpFunctions.h"

#include <list>

#include "astra/Globals.h"

#include "astra/AstraObjectManager.h"

#include "astra/Float32ProjectionData2D.h"
#include "astra/Float32VolumeData2D.h"
#include "astra/Float32ProjectionData3D.h"
#include "astra/Float32ProjectionData3DMemory.h"
#include "astra/Float32VolumeData3D.h"
#include "astra/Float32VolumeData3DMemory.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"

using namespace std;
using namespace astra;

#define USE_MATLAB_UNDOCUMENTED

//-----------------------------------------------------------------------------------------
/**
 * id = astra_mex_io_data('create', datatype, geometry, data);
 *        datatype: ['-vol','-sino','-sinocone'] 
 */
void astra_mex_data3d_create(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{ 
	// step1: get datatype
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	const mxArray * const geometry = prhs[2];
	const mxArray * const data = nrhs > 3 ? prhs[3] : NULL;

	if (!checkStructs(geometry)) {
		mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		return;
	}

	if (data && !checkDataType(data)) {
		mexErrMsgTxt("Data must be single, double or logical.");
		return;
	}

	const string sDataType = mex_util_get_string(prhs[1]);

	// step2: Allocate data
	CFloat32Data3DMemory* pDataObject3D =
			allocateDataObject(sDataType, geometry, data);
	if (!pDataObject3D) {
		// Error message was already set by the function
		return;
	}

	// step3: Initialize data
	if (!data) {
		mxArray * emptyArray = mxCreateDoubleMatrix(0, 0, mxREAL);
		copyMexToCFloat32Array(emptyArray, pDataObject3D->getData(),
				pDataObject3D->getSize());
		mxDestroyArray(emptyArray);
	} else {
		copyMexToCFloat32Array(data, pDataObject3D->getData(),
				pDataObject3D->getSize());
	}
	pDataObject3D->updateStatistics();

	// step4: store data object
	int iIndex = CData3DManager::getSingleton().store(pDataObject3D);

	// step5: return data id
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(iIndex);
	}
}

//-----------------------------------------------------------------------------------------

/** id = astra_mex_data3d('link', datatype, geometry, data);
 *   
 * Create a new data 3d object in the astra-library.
 * type: 
 * geom: MATLAB struct with the geometry for the data
 * data: A MATLAB matrix containing the data. 
 *       This matrix will be edited _in-place_!
 * id: identifier of the 3d data object as it is now stored in the astra-library.
 */

#ifdef USE_MATLAB_UNDOCUMENTED

void astra_mex_data3d_link(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{
	// TODO: Allow empty argument to let this function create its own mxArray
	// step1: get datatype
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	const mxArray * const geometry = prhs[2];
	const mxArray * const data = nrhs > 3 ? prhs[3] : NULL;
	const mxArray * const unshare = nrhs > 4 ? prhs[4] : NULL;
	const mxArray * const zIndex = nrhs > 5 ? prhs[5] : NULL;

	if (!checkStructs(geometry)) {
		mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		return;
	}

	if (data && !checkDataType(data)) {
		mexErrMsgTxt("Data must be single or double.");
		return;
	}

	string sDataType = mex_util_get_string(prhs[1]);

	// step2: Allocate data
	CFloat32Data3DMemory* pDataObject3D =
			allocateDataObject(sDataType, geometry, data, unshare, zIndex);
	if (!pDataObject3D) {
		// Error message was already set by the function
		return;
	}

	//pDataObject3D->updateStatistics();

	// step4: store data object
	int iIndex = CData3DManager::getSingleton().store(pDataObject3D);

	// step5: return data id
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(iIndex);
	}
}
#endif


//-----------------------------------------------------------------------------------------
/**
 * data = astra_mex_data3d('get', id);
 * 
 * Fetch data from the astra-library to a MATLAB matrix.
 * id: identifier of the 3d data object as stored in the astra-library.
 * data: MATLAB data

 */
void astra_mex_data3d_get(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	generic_astra_mex_data3d_get<mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
}

//-----------------------------------------------------------------------------------------
/**
 * data = astra_mex_data3d('get_single', id);
 * 
 * Fetch data from the astra-library to a MATLAB matrix.
 * id: identifier of the 3d data object as stored in the astra-library.
 * data: MATLAB data

 */
void astra_mex_data3d_get_single(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	generic_astra_mex_data3d_get<mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
}


//-----------------------------------------------------------------------------------------
/**
 * astra_mex_data3d('store', id, data);
 * 
 * Store MATLAB matrix data in the astra-library.
 * id: identifier of the 3d data object as stored in the astra-library.
 * data: MATLAB data

 */
void astra_mex_data3d_store(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: input
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	if (!checkDataType(prhs[2])) {
		mexErrMsgTxt("Data must be single or double.");
		return;
	}

	// step2: get data object
	CFloat32Data3DMemory* pDataObject = NULL;
	if (!checkID(mxGetScalar(prhs[1]), pDataObject)) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	copyMexToCFloat32Array(prhs[2], pDataObject->getData(), pDataObject->getSize());
	pDataObject->updateStatistics();
}

void astra_mex_data3d_dimensions(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: get input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iDid = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CFloat32Data3D* pData;
	if (!(pData = CData3DManager::getSingleton().get(iDid))) {
		mexErrMsgTxt("DataObject not valid. \n");
		return;
	}

	// step3: output
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(pData->getWidth());
	}	
	if (2 <= nlhs) {
		plhs[1] = mxCreateDoubleScalar(pData->getHeight());
	}	
	if (3 <= nlhs) {
		plhs[2] = mxCreateDoubleScalar(pData->getDepth());
	}	
}

//-----------------------------------------------------------------------------------------
/** geom = astra_mex_data3d('get_geometry', id);
 * 
 * Fetch the geometry of a 3d data object stored in the astra-library.
 * id: identifier of the 3d data object as stored in the astra-library.
 * geom: MATLAB-struct containing information about the used geometry.
 */
void astra_mex_data3d_get_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// parse input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value. \n");
		return;
	}
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// fetch data object
	CFloat32Data3D* pDataObject = astra::CData3DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		if (pDataObject->getType() == CFloat32Data3D::PROJECTION) {
			CFloat32ProjectionData3DMemory* pDataObject2 = dynamic_cast<CFloat32ProjectionData3DMemory*>(pDataObject);
			plhs[0] = createProjectionGeometryStruct(pDataObject2->getGeometry());
		}
		else if (pDataObject->getType() == CFloat32Data3D::VOLUME) {
			CFloat32VolumeData3DMemory* pDataObject2 = dynamic_cast<CFloat32VolumeData3DMemory*>(pDataObject);
			plhs[0] = createVolumeGeometryStruct(pDataObject2->getGeometry());
		}
	}


}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_data3d('delete', did1, did2, ...);
 */
void astra_mex_data3d_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: read input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	for (int i = 1; i < nrhs; i++) {
		int iDataID = (int)(mxGetScalar(prhs[i]));
		CData3DManager::getSingleton().remove(iDataID);
	}
}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_data3d('clear');
 */
void astra_mex_data3d_clear(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	CData3DManager::getSingleton().clear();
}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_data3d('info');
 */
void astra_mex_data3d_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	mexPrintf("%s", astra::CData3DManager::getSingleton().info().c_str());
}

//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: create, get, get_single, delete, clear, info\n");
	mexPrintf("             dimensions\n");
}


//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_io_data(mode,...);
 */
void mexFunction(int nlhs, mxArray* plhs[],
				 int nrhs, const mxArray* prhs[])
{

	// INPUT: Mode
	string sMode = "";
	if (1 <= nrhs) {
		sMode = mex_util_get_string(prhs[0]);	
	} else {
		printHelp();
		return;
	}

	// 3D data
	if (sMode ==  std::string("create")) { 
		astra_mex_data3d_create(nlhs, plhs, nrhs, prhs); 
#ifdef USE_MATLAB_UNDOCUMENTED
	} else if (sMode == "link") {
		astra_mex_data3d_link(nlhs, plhs, nrhs, prhs);
#endif
	} else if (sMode ==  std::string("get")) { 
		astra_mex_data3d_get(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("get_single")) { 
		astra_mex_data3d_get_single(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("store") ||
	           sMode ==  std::string("set")) { 
		astra_mex_data3d_store(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("delete")) { 
		astra_mex_data3d_delete(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == "clear") {
		astra_mex_data3d_clear(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "info") {
		astra_mex_data3d_info(nlhs, plhs, nrhs, prhs);
	} else if (sMode ==  std::string("dimensions")) { 
		astra_mex_data3d_dimensions(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("get_geometry")) {
		astra_mex_data3d_get_geometry(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}

	return;
}


