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

/** \file astra_mex_data3d_c.cpp
 *
 *  \brief Creates, manages and manipulates 3D volume and projection data objects.
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"
#include "mexCopyDataHelpFunctions.h"
#include "mexDataManagerHelpFunctions.h"

#include <list>

#include "astra/Globals.h"

#include "astra/AstraObjectManager.h"

#include "astra/Data3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"

using namespace std;
using namespace astra;

//-----------------------------------------------------------------------------------------
/**
 * id = astra_mex_io_data('create', datatype, geometry, data);
 *        datatype: ['-vol','-sino']
 */
void astra_mex_data3d_create(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{ 
	// step1: get datatype
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}

	const mxArray * const geometry = prhs[2];
	const mxArray * const data = nrhs > 3 ? prhs[3] : NULL;

	if (!checkStructs(geometry)) {
		mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.");
	}

	if (data && !checkDataType(data)) {
		mexErrMsgTxt("Data type must be single, double or logical.");
	}

	const string sDataType = mexToString(prhs[1]);

	// step2: Allocate data
	CData3D* pDataObject3D = allocateDataObject(sDataType, geometry, data);
	if (!pDataObject3D) {
		// Error message was already set by the function
		return;
	}

	// step3: Initialize data
	if (!data) {
		mxArray * emptyArray = mxCreateDoubleMatrix(0, 0, mxREAL);
		copyMexToCFloat32Array(emptyArray, pDataObject3D->getFloat32Memory(),
				pDataObject3D->getSize());
		mxDestroyArray(emptyArray);
	} else {
		copyMexToCFloat32Array(data, pDataObject3D->getFloat32Memory(),
				pDataObject3D->getSize());
	}

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
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}

	const mxArray * const geometry = prhs[2];
	const mxArray * const data = nrhs > 3 ? prhs[3] : NULL;
	const mxArray * const unshare = nrhs > 4 ? prhs[4] : NULL;
	const mxArray * const zIndex = nrhs > 5 ? prhs[5] : NULL;

	if (!checkStructs(geometry)) {
		mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.");
	}

	if (data && !mxIsSingle(data)) {
		mexErrMsgTxt("Data must be single.");
	}

	string sDataType = mexToString(prhs[1]);

	// step2: Allocate data
	CData3D* pDataObject3D = allocateDataObject(sDataType, geometry, data, unshare, zIndex);
	if (!pDataObject3D) {
		// Error message was already set by the function
		return;
	}

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
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}

	if (!checkDataType(prhs[2])) {
		mexErrMsgTxt("Data type must be single or double.");
	}

	// step2: get data object
	CData3D* pDataObject = NULL;
	if (!checkID(mxGetScalar(prhs[1]), pDataObject)) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
	}

	copyMexToCFloat32Array(prhs[2], pDataObject->getFloat32Memory(), pDataObject->getSize());
}

void astra_mex_data3d_dimensions(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: get input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}
	int iDid = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CData3D* pData;
	if (!(pData = CData3DManager::getSingleton().get(iDid))) {
		mexErrMsgTxt("Data object not found.");
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
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value.");
	}
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// fetch data object
	CData3D* pDataObject = astra::CData3DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.");
	}

	// create output
	if (1 <= nlhs) {
		if (pDataObject->getType() == CData3D::PROJECTION) {
			CFloat32ProjectionData3D* pDataObject2 = dynamic_cast<CFloat32ProjectionData3D*>(pDataObject);
			plhs[0] = configToStruct(pDataObject2->getGeometry()->getConfiguration());
			
		}
		else if (pDataObject->getType() == CData3D::VOLUME) {
			CFloat32VolumeData3D* pDataObject2 = dynamic_cast<CFloat32VolumeData3D*>(pDataObject);
			plhs[0] = configToStruct(pDataObject2->getGeometry()->getConfiguration());
		}
	}


}

//-----------------------------------------------------------------------------------------
/** astra_mex_data3d('change_geometry', id, geom);
 *
 * Change the geometry of a 3d data object.
 * id: identifier of the 3d data object as stored in the astra-library.
 * geom: the new geometry struct, as created by astra_create_vol/proj_geom
 */
void astra_mex_data3d_change_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// parse input
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}

	// get data object
	CData3D* pDataObject = NULL;
	if (!checkID(mxGetScalar(prhs[1]), pDataObject)) {
		mexErrMsgTxt("Data object not found or not initialized properly.");
	}

	const mxArray * const geometry = prhs[2];

	if (!checkStructs(geometry)) {
		mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.");
	}

	CFloat32ProjectionData3D* pProjData = dynamic_cast<CFloat32ProjectionData3D*>(pDataObject);
	if (pProjData) {
		// Projection data

		// Read geometry
		astra::Config* cfg = structToConfig("ProjectionGeometry3D", geometry);
		// FIXME: Change how the base class is created. (This is duplicated
		// in Projector3D.cpp.)
		std::string type = cfg->self.getAttribute("type");
		astra::CProjectionGeometry3D* pGeometry = 0;
		if (type == "parallel3d") {
			pGeometry = new astra::CParallelProjectionGeometry3D();
		} else if (type == "parallel3d_vec") {
			pGeometry = new astra::CParallelVecProjectionGeometry3D();
		} else if (type == "cone") {
			pGeometry = new astra::CConeProjectionGeometry3D();
		} else if (type == "cone_vec") {
			pGeometry = new astra::CConeVecProjectionGeometry3D();
		} else {
			std::string message = "'" + type + "' is not a valid 3D geometry type.";
			mexErrMsgTxt(message.c_str());
		}

		if (!pGeometry->initialize(*cfg)) {
			delete pGeometry;
			delete cfg;
			mexErrMsgWithAstraLog("Geometry class could not be initialized.");
		}
		delete cfg;

		// Check dimensions
		if (pGeometry->getDetectorColCount() != pProjData->getDetectorColCount() ||
		    pGeometry->getProjectionCount() != pProjData->getAngleCount() ||
		    pGeometry->getDetectorRowCount() != pProjData->getDetectorRowCount())
		{
			delete pGeometry;
			mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
		}

		// If ok, change geometry
		pProjData->changeGeometry(pGeometry);
		delete pGeometry;
	} else {
		// Volume data
		CFloat32VolumeData3D* pVolData = dynamic_cast<CFloat32VolumeData3D*>(pDataObject);
		assert(pVolData);

		// Read geometry
		astra::Config* cfg = structToConfig("VolumeGeometry3D", geometry);
		astra::CVolumeGeometry3D* pGeometry = new astra::CVolumeGeometry3D();
		if (!pGeometry->initialize(*cfg))
		{
			delete pGeometry;
			delete cfg;
			mexErrMsgWithAstraLog("Geometry class could not be initialized.");
		}
		delete cfg;

				// Check dimensions
		if (pGeometry->getGridColCount() != pVolData->getColCount() ||
		    pGeometry->getGridRowCount() != pVolData->getRowCount() ||
		    pGeometry->getGridSliceCount() != pVolData->getSliceCount())
		{
			delete pGeometry;
			mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
		}

		// If ok, change geometry
		pVolData->changeGeometry(pGeometry);
		delete pGeometry;
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
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
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
	mexPrintf("             dimensions, get_geometry, change_geometry\n");
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
		sMode = mexToString(prhs[0]);	
	} else {
		printHelp();
		return;
	}

	initASTRAMex();

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
	} else if (sMode == std::string("change_geometry")) {
		astra_mex_data3d_change_geometry(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}

	return;
}


