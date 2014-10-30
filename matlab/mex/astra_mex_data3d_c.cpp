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
		mexErrMsgTxt("Data must be single or double.");
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
 * [id] = astra_mex_io_data('create_cache', config);
 */
void astra_mex_data3d_create_cache(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
//	if (nrhs < 2) {
//		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
//		return;
//	}
//
//	if (!mxIsStruct(prhs[1])) {
//		mexErrMsgTxt("Argument 1 not a valid MATLAB struct. \n");
//	}
//
//	// turn MATLAB struct to an XML-based Config object
//	XMLDocument* xml = struct2XML("Data3D", prhs[1]);
//	Config cfg;
//	cfg.self = xml->getRootNode();
//
//	// create dataobject
//	string sType = cfg.self->getAttribute("type");
//	int iIndex;
//	if (sType == "ProjectionCached") {
//		CFloat32ProjectionData3DCached* pData = new CFloat32ProjectionData3DCached(cfg);
//		iIndex = CData3DManager::getSingleton().store(pData);
//	}
////	else if (sType == "VolumeCached") {
////		CFloat32VolumeData3DCached* pData = new CFloat32VolumeData3DCached(cfg);
////		pData->initialize(cfg);
////		iIndex = CData3DManager::getSingleton().store(pData);
////	}
//
//	// step4: set output
//	if (1 <= nlhs) {
//		plhs[0] = mxCreateDoubleScalar(iIndex);
//	}

}


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


//-----------------------------------------------------------------------------------------
/**
 * [id] = astra_mex_io_data('fetch_slice', id, slicenr);
 */
void astra_mex_data3d_fetch_slice_z(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
//	// step1: get input
//	if (nrhs < 3) {
//		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
//		return;
//	}
//	int iDid = (int)(mxGetScalar(prhs[1]));
//	int iSliceNr = (int)(mxGetScalar(prhs[2]));
//
//	// Get data object
//	CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
//	if (!pData) {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;
//	}
//
//	CFloat32Data2D* res = NULL;
//	// Projection Data
//	if (pData->getType() == CFloat32Data3D::PROJECTION) {
//		CFloat32ProjectionData3D* pData2 = dynamic_cast<CFloat32ProjectionData3D*>(pData);
////		res = pData2->fetchSlice(iSliceNr);
//	} 
//	// Volume Data
//	else if (pData->getType() == CFloat32Data3D::VOLUME) {
//		CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
////		res = pData2->fetchSliceZ(iSliceNr);
//	} 
//	// Error
//	else {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;	
//	}
//	
//	// store data
//	int iIndex = CData2DManager::getSingleton().store(res);
//
//	// step4: set output
//	if (1 <= nlhs) {
//		plhs[0] = mxCreateDoubleScalar(iIndex);
//	}
}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_io_data('returnSlice', id, slicenr);
 */
void astra_mex_data3d_return_slice_z(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
//	// step1: get input
//	if (nrhs < 3) {
//		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
//		return;
//	}
//	int iDid = (int)(mxGetScalar(prhs[1]));
//	int iSliceNr = (int)(mxGetScalar(prhs[2]));
//
//	// Get data object
//	CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
//	if (!pData) {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;
//	}
//
//	// Projection Data
//	if (pData->getType() == CFloat32Data3D::PROJECTION) {
//		CFloat32ProjectionData3D* pData2 = dynamic_cast<CFloat32ProjectionData3D*>(pData);
//// TODO: think about returning slices
////		pData2->returnSlice(iSliceNr);
//	} 
//	// Volume Data
//	else if (pData->getType() == CFloat32Data3D::VOLUME) {
//		CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
//// TODO: think about returning slices
////      	pData2->returnSliceZ(iSliceNr);
//	} 
//	// Error
//	else {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;	
//	}
}

//-----------------------------------------------------------------------------------------
/**
 * [id] = astra_mex_io_data('fetch_projection', id, slicenr);
 */
void astra_mex_data3d_fetch_projection(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//// step1: get input
	//if (nrhs < 3) {
	//	mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
	//	return;
	//}
	//int iDid = (int)(mxGetScalar(prhs[1]));
	//int iProjectionNr = (int)(mxGetScalar(prhs[2]));

	//// Get data object
	//CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
	//if (!pData) {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;
	//}

	//CFloat32Data2D* res = NULL;
	//// Projection Data
	//if (pData->getType() == CFloat32Data3D::PROJECTION) {
	//	CFloat32ProjectionData3D* pData2 = dynamic_cast<CFloat32ProjectionData3D*>(pData);
	//	res = pData2->fetchProjection(iProjectionNr);
	//} 
	//// Error
	//else {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;	
	//}
	//
	//// store data
	//int iIndex = CData2DManager::getSingleton().store(res);

	//// step4: set output
	//if (1 <= nlhs) {
	//	plhs[0] = mxCreateDoubleScalar(iIndex);
	//}
}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_io_data('return_projection', id, slicenr);
 */
void astra_mex_data3d_return_projection(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//// step1: get input
	//if (nrhs < 3) {
	//	mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
	//	return;
	//}
	//int iDid = (int)(mxGetScalar(prhs[1]));
	//int iProjectionNr = (int)(mxGetScalar(prhs[2]));

	//// Get data object
	//CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
	//if (!pData) {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;
	//}

	//// Projection Data
	//if (pData->getType() == CFloat32Data3D::PROJECTION) {
	//	CFloat32ProjectionData3D* pData2 = dynamic_cast<CFloat32ProjectionData3D*>(pData);
	////	pData2->returnProjection(iProjectionNr);
	//} 
	//// Error
	//else {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;	
	//}
}

//-----------------------------------------------------------------------------------------
/**
 * [id] = astra_mex_io_data('fetch_projection', id, slicenr);
 */
void astra_mex_data3d_fetch_slice_x(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//// step1: get input
	//if (nrhs < 3) {
	//	mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
	//	return;
	//}
	//int iDid = (int)(mxGetScalar(prhs[1]));
	//int iSliceNr = (int)(mxGetScalar(prhs[2]));

	//// Get data object
	//CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
	//if (!pData) {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;
	//}

	//CFloat32Data2D* res = NULL;
	//// Projection Data
	//if (pData->getType() == CFloat32Data3D::VOLUME) {
	//	CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
	//	res = pData2->fetchSliceX(iSliceNr);
	//} 
	//// Error
	//else {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;	
	//}
	//
	//// store data
	//int iIndex = CData2DManager::getSingleton().store(res);

	//// step4: set output
	//if (1 <= nlhs) {
	//	plhs[0] = mxCreateDoubleScalar(iIndex);
	//}
}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_io_data('return_slice_x', id, slicenr);
 */
void astra_mex_data3d_return_slice_x(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
//	// step1: get input
//	if (nrhs < 3) {
//		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
//		return;
//	}
//	int iDid = (int)(mxGetScalar(prhs[1]));
//	int iSliceNr = (int)(mxGetScalar(prhs[2]));
//
//	// Get data object
//	CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
//	if (!pData) {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;
//	}
//
//	// Projection Data
//	if (pData->getType() == CFloat32Data3D::VOLUME) {
//		CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
//// TODO: think about returning slices
////		pData2->returnSliceX(iSliceNr);
//	} 
//	// Error
//	else {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;	
//	}
}


//-----------------------------------------------------------------------------------------
/**
 * [id] = astra_mex_io_data('fetch_slice_y', id, slicenr);
 */
void astra_mex_data3d_fetch_slice_y(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//// step1: get input
	//if (nrhs < 3) {
	//	mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
	//	return;
	//}
	//int iDid = (int)(mxGetScalar(prhs[1]));
	//int iSliceNr = (int)(mxGetScalar(prhs[2]));

	//// Get data object
	//CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
	//if (!pData) {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;
	//}

	//CFloat32Data2D* res = NULL;
	//// Projection Data
	//if (pData->getType() == CFloat32Data3D::VOLUME) {
	//	CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
	//	res = pData2->fetchSliceY(iSliceNr);
	//} 
	//// Error
	//else {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;	
	//}
	//
	//// store data
	//int iIndex = CData2DManager::getSingleton().store(res);

	//// step4: set output
	//if (1 <= nlhs) {
	//	plhs[0] = mxCreateDoubleScalar(iIndex);
	//}
}

//-----------------------------------------------------------------------------------------
/**
 * astra_mex_io_data('return_slice_y', id, slicenr);
 */
void astra_mex_data3d_return_slice_y(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
//	// step1: get input
//	if (nrhs < 3) {
//		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
//		return;
//	}
//	int iDid = (int)(mxGetScalar(prhs[1]));
//	int iSliceNr = (int)(mxGetScalar(prhs[2]));
//
//	// Get data object
//	CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
//	if (!pData) {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;
//	}
//
//	// Projection Data
//	if (pData->getType() == CFloat32Data3D::VOLUME) {
//		CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
//// TODO: think about returning slices
////		pData2->returnSliceY(iSliceNr);
//	} 
//	// Error
//	else {
//		mexErrMsgTxt("DataObject not valid. \n");
//		return;	
//	}
}

//-----------------------------------------------------------------------------------------
/**
 * [dim_x dim_y dim_z] = astra_mex_io_data('dimensions', id);
 */
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
/**
 * [geom] = astra_mex_data3d('geometry', id);
 */
void astra_mex_data3d_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	//// Get input
	//if (nrhs < 2) {
	//	mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
	//	return;
	//}
	//int iDid = (int)(mxGetScalar(prhs[1]));

	//// Get data object
	//CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
	//if (!pData) {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;
	//}

	//// Projection Data
	//if (pData->getType() == CFloat32Data3D::PROJECTION) {
	//	CFloat32ProjectionData3D* pData2 = dynamic_cast<CFloat32ProjectionData3D*>(pData);
	//	CProjectionGeometry3D* pProjGeom = pData2->getGeometry();
	//	XMLDocument* config = pProjGeom->toXML();

	//	if (1 <= nlhs) {
	//		plhs[0] = XML2struct(config);
	//	}
	//} 
	//// Volume Data
	//else if (pData->getType() == CFloat32Data3D::VOLUME) {
	////	CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
	////	CVolumeGeometry2D* pVolGeom = pData2->getGeometry2D(iSliceNr);
	////	if (1 <= nlhs) {
	////		plhs[0] = createVolumeGeometryStruct(pVolGeom);
	////	}
	//} 
	//// Error
	//else {
	//	mexErrMsgTxt("Type not valid. \n");
	//	return;	
	//}
}

//-----------------------------------------------------------------------------------------
/**
 * [geom_xml] = astra_mex_data3d('geometry_xml', id);
 */
void astra_mex_data3d_geometry_xml(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	//// Get input
	//if (nrhs < 2) {
	//	mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
	//	return;
	//}
	//int iDid = (int)(mxGetScalar(prhs[1]));

	//// Get data object
	//CFloat32Data3D* pData = CData3DManager::getSingleton().get(iDid);
	//if (!pData) {
	//	mexErrMsgTxt("DataObject not valid. \n");
	//	return;
	//}

	//// Projection Data
	//if (pData->getType() == CFloat32Data3D::PROJECTION) {
	//	CFloat32ProjectionData3D* pData2 = dynamic_cast<CFloat32ProjectionData3D*>(pData);
	//	CProjectionGeometry3D* pProjGeom = pData2->getGeometry();
	//	XMLDocument* config = pProjGeom->toXML();

	//	if (1 <= nlhs) {
	//		plhs[0] = mxCreateString(config->getRootNode()->toString().c_str());
	//	}
	//} 
	//// Volume Data
	//else if (pData->getType() == CFloat32Data3D::VOLUME) {
	////	CFloat32VolumeData3D* pData2 = dynamic_cast<CFloat32VolumeData3D*>(pData);
	////	CVolumeGeometry2D* pVolGeom = pData2->getGeometry2D(iSliceNr);
	////	if (1 <= nlhs) {
	////		plhs[0] = createVolumeGeometryStruct(pVolGeom);
	////	}
	//} 
	//// Error
	//else {
	//	mexErrMsgTxt("Type not valid. \n");
	//	return;	
	//}
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
	mexPrintf("Valid modes: create, create_cache, get, get_single, delete, clear, info\n");
	mexPrintf("             fetch_projection, return_projection, fetch_slice[_z],\n");
	mexPrintf("             return_slice[_z], fetch_slice_x, return slice_x\n");
	mexPrintf("             fetch_slice_y, return slice_y, dimensions, geometry\n");
	mexPrintf("             geometry_xml\n");
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
	} else if (sMode ==  std::string("create_cache")) { 
		astra_mex_data3d_create_cache(nlhs, plhs, nrhs, prhs); 
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
	} else if (sMode ==  std::string("fetch_projection")) { 
		astra_mex_data3d_fetch_projection(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("return_projection")) { 
		astra_mex_data3d_return_projection(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("fetch_slice") || sMode ==  std::string("fetch_slice_z")) { 
		astra_mex_data3d_fetch_slice_z(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("return_slice")  || sMode ==  std::string("return_slice_z")) { 
		astra_mex_data3d_return_slice_z(nlhs, plhs, nrhs, prhs);
	} else if (sMode ==  std::string("fetch_slice_x")) { 
		astra_mex_data3d_fetch_slice_x(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("return_slice_x")) { 
		astra_mex_data3d_return_slice_x(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("fetch_slice_y")) { 
		astra_mex_data3d_fetch_slice_y(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("return_slice_y")) { 
		astra_mex_data3d_return_slice_y(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("dimensions")) { 
		astra_mex_data3d_dimensions(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("geometry")) {
		astra_mex_data3d_geometry(nlhs, plhs, nrhs, prhs);
	} else if (sMode == std::string("geometry_xml")) {
		astra_mex_data3d_geometry_xml(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}

	return;
}


