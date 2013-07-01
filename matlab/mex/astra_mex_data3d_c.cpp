/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

	string sDataType = mex_util_get_string(prhs[1]);	
	CFloat32Data3DMemory* pDataObject3D = NULL;

	if (nrhs >= 4 && !(mex_is_scalar(prhs[3]) || mxIsDouble(prhs[3]) || mxIsSingle(prhs[3]))) {
		mexErrMsgTxt("Data must be single or double.");
		return;
	}

	mwSize dims[3];

	// SWITCH DataType
	if (sDataType == "-vol") {

		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		Config cfg;
		XMLDocument* xml = struct2XML("VolumeGeometry", prhs[2]);
		if (!xml)
			return;
		cfg.self = xml->getRootNode();
		CVolumeGeometry3D* pGeometry = new CVolumeGeometry3D();
		if (!pGeometry->initialize(cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete pGeometry;
			delete xml;
			return;
		}
		delete xml;

		// If data is specified, check dimensions
		if (nrhs >= 4 && !mex_is_scalar(prhs[3])) {
			get3DMatrixDims(prhs[3], dims);
			if (pGeometry->getGridColCount() != dims[0] || pGeometry->getGridRowCount() != dims[1] || pGeometry->getGridSliceCount() != dims[2]) {
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete pGeometry;
				return;
			}
		}

		// Initialize data object
		pDataObject3D = new CFloat32VolumeData3DMemory(pGeometry);		
		delete pGeometry;
	}

	else if (sDataType == "-sino" || sDataType == "-proj3d") {

		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		XMLDocument* xml = struct2XML("ProjectionGeometry", prhs[2]);
		if (!xml)
			return;
		Config cfg;
		cfg.self = xml->getRootNode();

		// FIXME: Change how the base class is created. (This is duplicated
		// in Projector2D.cpp.)
		std::string type = cfg.self->getAttribute("type");
		CProjectionGeometry3D* pGeometry = 0;
		if (type == "parallel3d") {
			pGeometry = new CParallelProjectionGeometry3D();
		} else if (type == "parallel3d_vec") {
			pGeometry = new CParallelVecProjectionGeometry3D();
		} else if (type == "cone") {
			pGeometry = new CConeProjectionGeometry3D();
		} else if (type == "cone_vec") {
			pGeometry = new CConeVecProjectionGeometry3D();
		} else {
			mexErrMsgTxt("Invalid geometry type.\n");
			return;
		}

		if (!pGeometry->initialize(cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete pGeometry;
			delete xml;
			return;
		}
		delete xml;

		// If data is specified, check dimensions
		if (nrhs >= 4 && !mex_is_scalar(prhs[3])) {
			get3DMatrixDims(prhs[3], dims);
			if (pGeometry->getDetectorColCount() != dims[0] || pGeometry->getProjectionCount() != dims[1] || pGeometry->getDetectorRowCount() != dims[2]) {
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete pGeometry;
				return;
			}
		}

		// Initialize data object
		pDataObject3D = new CFloat32ProjectionData3DMemory(pGeometry);		
	}

	else if (sDataType == "-sinocone") {
		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		XMLDocument* xml = struct2XML("ProjectionGeometry", prhs[2]);
		if (!xml)
			return;
		Config cfg;
		cfg.self = xml->getRootNode();
		CConeProjectionGeometry3D* pGeometry = new CConeProjectionGeometry3D();
		if (!pGeometry->initialize(cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete xml;
			delete pGeometry;
			return;
		}
		delete xml;
		// If data is specified, check dimensions
		if (nrhs >= 4 && !mex_is_scalar(prhs[3])) {
			get3DMatrixDims(prhs[3], dims);
			if (pGeometry->getDetectorRowCount() != dims[2] || pGeometry->getProjectionCount() != dims[1] || pGeometry->getDetectorColCount() != dims[0]) {
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete pGeometry;
				return;
			}
		}
		// Initialize data object
		pDataObject3D = new CFloat32ProjectionData3DMemory(pGeometry);		
		delete pGeometry;
	}
	else {
		mexErrMsgTxt("Invalid datatype.  Please specify '-vol' or '-proj3d'. \n");
		return;
	}

	// Check initialization
	if (!pDataObject3D->isInitialized()) {
		mexErrMsgTxt("Couldn't initialize data object.\n");
		delete pDataObject3D;
		return;
	}

	// Store data

	// fill with scalar value
	if (nrhs < 4 || mex_is_scalar(prhs[3])) {
		float32 fValue = 0.0f;
		if (nrhs >= 4)
			fValue = (float32)mxGetScalar(prhs[3]);
		for (int i = 0; i < pDataObject3D->getSize(); ++i) {
			pDataObject3D->getData()[i] = fValue;
		}
	}
	// fill with array value
	else if (mxIsDouble(prhs[3])) {
		double* pdMatlabData = mxGetPr(prhs[3]);
		int i = 0;
		int col, row, slice;
		for (slice = 0; slice < dims[2]; ++slice) {
			for (row = 0; row < dims[1]; ++row) {
				for (col = 0; col < dims[0]; ++col) {
					// TODO: Benchmark and remove triple indexing?
					pDataObject3D->getData3D()[slice][row][col] = pdMatlabData[i];
					++i;
				}
			}
		}
	}
	else if (mxIsSingle(prhs[3])) {
		const float* pfMatlabData = (const float*)mxGetData(prhs[3]);
		int i = 0;
		int col, row, slice;
		for (slice = 0; slice < dims[2]; ++slice) {
			for (row = 0; row < dims[1]; ++row) {
				for (col = 0; col < dims[0]; ++col) {
					// TODO: Benchmark and remove triple indexing?
					pDataObject3D->getData3D()[slice][row][col] = pfMatlabData[i];
					++i;
				}
			}
		}
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
	// step1: input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CFloat32Data3DMemory* pDataObject = dynamic_cast<CFloat32Data3DMemory*>(astra::CData3DManager::getSingleton().get(iDataID));
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		mwSize dims[3];
		dims[0] = pDataObject->getWidth();
		dims[1] = pDataObject->getHeight();
		dims[2] = pDataObject->getDepth();

		plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
		double* out = mxGetPr(plhs[0]);

		int i = 0;
		for (int slice = 0; slice < pDataObject->getDepth(); slice++) {
			for (int row = 0; row < pDataObject->getHeight(); row++) {
				for (int col = 0; col < pDataObject->getWidth(); col++) {
					// TODO: Benchmark and remove triple indexing?
					out[i] = pDataObject->getData3D()[slice][row][col];
					++i;
				}
			}
		}	
	}
	
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
	// step1: input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CFloat32Data3DMemory* pDataObject = dynamic_cast<CFloat32Data3DMemory*>(astra::CData3DManager::getSingleton().get(iDataID));
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		mwSize dims[3];
		dims[0] = pDataObject->getWidth();
		dims[1] = pDataObject->getHeight();
		dims[2] = pDataObject->getDepth();

		plhs[0] = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
		float* out = (float *)mxGetData(plhs[0]);

		int i = 0;
		for (int slice = 0; slice < pDataObject->getDepth(); slice++) {
			for (int row = 0; row < pDataObject->getHeight(); row++) {
				for (int col = 0; col < pDataObject->getWidth(); col++) {
					// TODO: Benchmark and remove triple indexing?
					out[i] = pDataObject->getData3D()[slice][row][col];
					++i;
				}
			}
		}	
	}
	
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
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CFloat32Data3DMemory* pDataObject = dynamic_cast<CFloat32Data3DMemory*>(astra::CData3DManager::getSingleton().get(iDataID));
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	if (!(mex_is_scalar(prhs[2]) || mxIsDouble(prhs[2]) || mxIsSingle(prhs[2]))) {
		mexErrMsgTxt("Data must be single or double.");
		return;
	}

	// fill with scalar value
	if (mex_is_scalar(prhs[2])) {
		float32 fValue = (float32)mxGetScalar(prhs[2]);
		for (int i = 0; i < pDataObject->getSize(); ++i) {
			pDataObject->getData()[i] = fValue;
		}
	}
	// fill with array value
	else if (mxIsDouble(prhs[2])) {
		mwSize dims[3];
		get3DMatrixDims(prhs[2], dims);
		if (dims[0] != pDataObject->getWidth() || dims[1] != pDataObject->getHeight() || dims[2] != pDataObject->getDepth()) {
			mexErrMsgTxt("Data object dimensions don't match.\n");
			return;

		}
		double* pdMatlabData = mxGetPr(prhs[2]);
		int i = 0;
		int col, row, slice;
		for (slice = 0; slice < dims[2]; ++slice) {
			for (row = 0; row < dims[1]; ++row) {
				for (col = 0; col < dims[0]; ++col) {
					// TODO: Benchmark and remove triple indexing?
					pDataObject->getData3D()[slice][row][col] = pdMatlabData[i];
					++i;
				}
			}
		}
	}
	else if (mxIsSingle(prhs[2])) {
		mwSize dims[3];
		get3DMatrixDims(prhs[2], dims);
		if (dims[0] != pDataObject->getWidth() || dims[1] != pDataObject->getHeight() || dims[2] != pDataObject->getDepth()) {
			mexErrMsgTxt("Data object dimensions don't match.\n");
			return;

		}
		const float* pfMatlabData = (const float *)mxGetData(prhs[2]);
		int i = 0;
		int col, row, slice;
		for (slice = 0; slice < dims[2]; ++slice) {
			for (row = 0; row < dims[1]; ++row) {
				for (col = 0; col < dims[0]; ++col) {
					// TODO: Benchmark and remove triple indexing?
					pDataObject->getData3D()[slice][row][col] = pfMatlabData[i];
					++i;
				}
			}
		}
	}
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


