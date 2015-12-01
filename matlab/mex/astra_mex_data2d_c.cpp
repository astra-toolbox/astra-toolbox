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

/** \file astra_mex_data2d_c.cpp
 *
 *  \brief Creates, manages and manipulates 2D volume and projection data objects.
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include <list>

#include "astra/Globals.h"

#include "astra/AstraObjectManager.h"

#include "astra/Float32ProjectionData2D.h"
#include "astra/Float32VolumeData2D.h"
#include "astra/SparseMatrixProjectionGeometry2D.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"

using namespace std;
using namespace astra;

//-----------------------------------------------------------------------------------------
/** astra_mex_data2d('delete', id1, id2, ...);
 *
 * Delete one or more data objects currently stored in the astra-library.
 * id1, id2, ... : identifiers of the 2d data objects as stored in the astra-library.
 */
void astra_mex_data2d_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: read input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	// step2: delete all specified data objects
	for (int i = 1; i < nrhs; i++) {
		int iDataID = (int)(mxGetScalar(prhs[i]));
		CData2DManager::getSingleton().remove(iDataID);
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_data2d('clear');
 *
 * Delete all data objects currently stored in the astra-library.
 */
void astra_mex_data2d_clear(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	CData2DManager::getSingleton().clear();
}

//-----------------------------------------------------------------------------------------
/** id = astra_mex_data2d('create', datatype, geometry, data);
 *   
 * Create a new data 2d object in the astra-library.
 * type: '-vol' for volume data, '-sino' for projection data
 * geom: MATLAB struct with the geometry for the data
 * data: Optional. Can be either a MATLAB matrix containing the data. In that case the dimensions 
 * should match that of the geometry of the object.  It can also be a single value, in which case 
 * the entire data will be set to that value.  If this isn't specified all values are set to 0.
 * id: identifier of the 2d data object as it is now stored in the astra-library.
 */
void astra_mex_data2d_create(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{ 
	// step1: get datatype
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	string sDataType = mexToString(prhs[1]);	
	CFloat32Data2D* pDataObject2D = NULL;

	if (nrhs >= 4 && !(mexIsScalar(prhs[3])|| mxIsDouble(prhs[3]) || mxIsLogical(prhs[3]) || mxIsSingle(prhs[3]) )) {
		mexErrMsgTxt("Data must be single, double or logical.");
		return;
	}
	if (mxIsSparse(prhs[2])) {
		mexErrMsgTxt("Data may not be sparse.");
		return;
	}

	// SWITCH DataType
	if (sDataType == "-vol") {
		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		
		Config* cfg = structToConfig("VolumeGeometry", prhs[2]);
		CVolumeGeometry2D* pGeometry = new CVolumeGeometry2D();
		if (!pGeometry->initialize(*cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete cfg;
			delete pGeometry;
			return;
		}
		// If data is specified, check dimensions
		if (nrhs >= 4 && !mexIsScalar(prhs[3])) {
			if (pGeometry->getGridColCount() != mxGetN(prhs[3]) || pGeometry->getGridRowCount() != mxGetM(prhs[3])) {
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete cfg;
				delete pGeometry;
				return;
			}
		}
		// Initialize data object
		pDataObject2D = new CFloat32VolumeData2D(pGeometry);		
		delete pGeometry;
		delete cfg;
	}
	else if (sDataType == "-sino") {
		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		
		Config* cfg = structToConfig("ProjectionGeometry", prhs[2]);
		// FIXME: Change how the base class is created. (This is duplicated
		// in 'change_geometry' and Projector2D.cpp.)
		std::string type = cfg->self.getAttribute("type");
		CProjectionGeometry2D* pGeometry;
		if (type == "sparse_matrix") {
			pGeometry = new CSparseMatrixProjectionGeometry2D();
		} else if (type == "fanflat") {
			//CFanFlatProjectionGeometry2D* pFanFlatProjectionGeometry = new CFanFlatProjectionGeometry2D();
			//pFanFlatProjectionGeometry->initialize(Config(node));
			//m_pProjectionGeometry = pFanFlatProjectionGeometry;
			pGeometry = new CFanFlatProjectionGeometry2D();	
		} else if (type == "fanflat_vec") {
			pGeometry = new CFanFlatVecProjectionGeometry2D();	
		} else {
			pGeometry = new CParallelProjectionGeometry2D();	
		}
		if (!pGeometry->initialize(*cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete pGeometry;
			delete cfg;
			return;
		}
		// If data is specified, check dimensions
		if (nrhs >= 4 && !mexIsScalar(prhs[3])) {
			if (pGeometry->getDetectorCount() != mxGetN(prhs[3]) || pGeometry->getProjectionAngleCount() != mxGetM(prhs[3])) {
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete pGeometry;
				delete cfg;
				return;
			}
		}
		// Initialize data object
		pDataObject2D = new CFloat32ProjectionData2D(pGeometry);
		delete pGeometry;
		delete cfg;
	}
	else {
		mexErrMsgTxt("Invalid datatype.  Please specify '-vol' or '-sino'. \n");
		return;
	}

	// Check initialization
	if (!pDataObject2D->isInitialized()) {
		mexErrMsgTxt("Couldn't initialize data object.\n");
		delete pDataObject2D;
		return;
	}

	// Store data
	if (nrhs == 3) {
		for (int i = 0; i < pDataObject2D->getSize(); ++i) {
			pDataObject2D->getData()[i] = 0.0f;
		}
	}

	// Store data
	if (nrhs >= 4) {
		// fill with scalar value
		if (mexIsScalar(prhs[3])) {
			float32 fValue = (float32)mxGetScalar(prhs[3]);
			for (int i = 0; i < pDataObject2D->getSize(); ++i) {
				pDataObject2D->getData()[i] = fValue;
			}
		}
		// fill with array value
		else {
			const mwSize* dims = mxGetDimensions(prhs[3]);
			// Check Data dimensions
			if (pDataObject2D->getWidth() != mxGetN(prhs[3]) || pDataObject2D->getHeight() != mxGetM(prhs[3])) {
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				return;
			}

			// logical data		
			if (mxIsLogical(prhs[3])) {
				mxLogical* pbMatlabData = mxGetLogicals(prhs[3]);
				int i = 0;
				int col, row;
				for (col = 0; col < dims[1]; ++col) {
					for (row = 0; row < dims[0]; ++row) {
						pDataObject2D->getData2D()[row][col] = (float32)pbMatlabData[i];
						++i;
					}
				}
			// double data
			} else if (mxIsDouble(prhs[3])) {
				double* pdMatlabData = mxGetPr(prhs[3]);
				int i = 0;
				int col, row;
				for (col = 0; col < dims[1]; ++col) {
					for (row = 0; row < dims[0]; ++row) {
						pDataObject2D->getData2D()[row][col] = pdMatlabData[i];
						++i;
					}
				}
			// single data
			} else if (mxIsSingle(prhs[3])) {
				const float* pfMatlabData = (const float *)mxGetData(prhs[3]);
				int i = 0;
				int col, row;
				for (col = 0; col < dims[1]; ++col) {
					for (row = 0; row < dims[0]; ++row) {
						pDataObject2D->getData2D()[row][col] = pfMatlabData[i];
						++i;
					}
				}
			} else {
				ASTRA_ASSERT(false);
			}
		}
	}

	// step4: store data object
	int iIndex = CData2DManager::getSingleton().store(pDataObject2D);

	// step5: return data id
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(iIndex);
	}

}

//-----------------------------------------------------------------------------------------
/** astra_mex_data2d('store', id, data);
 *
 * Store data in an existing astra 2d dataobject with a MATLAB matrix or with a scalar value. 
 * id: identifier of the 2d data object as stored in the astra-library.
 * data: can be either a MATLAB matrix containing the data. In that case the dimensions should match that of the geometry of the object.  It can also be a single value, in which case the entire data will be set to that value.
 */
void astra_mex_data2d_store(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: input
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value. \n");
		return;
	}
	int iDataID = (int)(mxGetScalar(prhs[1]));

	if (!(mexIsScalar(prhs[2]) || mxIsDouble(prhs[2]) || mxIsLogical(prhs[2]) || mxIsSingle(prhs[2]))) {
		mexErrMsgTxt("Data must be single, double or logical.");
		return;
	}
	if (mxIsSparse(prhs[2])) {
		mexErrMsgTxt("Data may not be sparse.");
		return;
	}

	// step2: get data object
	CFloat32Data2D* pDataObject = astra::CData2DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}
	
	// step3: insert data
	// fill with scalar value
	if (mexIsScalar(prhs[2])) {
		float32 fValue = (float32)mxGetScalar(prhs[2]);
		for (int i = 0; i < pDataObject->getSize(); ++i) {
			pDataObject->getData()[i] = fValue;
		}
	} else {
		// Check Data dimensions
		if (pDataObject->getWidth() != mxGetN(prhs[2]) || pDataObject->getHeight() != mxGetM(prhs[2])) {
			mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
			return;
		}
		const mwSize* dims = mxGetDimensions(prhs[2]);

		// logical data		
		if (mxIsLogical(prhs[2])) {
			mxLogical* pbMatlabData = mxGetLogicals(prhs[2]);
			int i = 0;
			int col, row;
			for (col = 0; col < dims[1]; ++col) {
				for (row = 0; row < dims[0]; ++row) {
					pDataObject->getData2D()[row][col] = (float32)pbMatlabData[i];
					++i;
				}
			}
		// double data
		} else if (mxIsDouble(prhs[2])) {
			double* pdMatlabData = mxGetPr(prhs[2]);
			int i = 0;
			int col, row;
			for (col = 0; col < dims[1]; ++col) {
				for (row = 0; row < dims[0]; ++row) {
					pDataObject->getData2D()[row][col] = pdMatlabData[i];
					++i;
				}
			}
		// single data
		} else if (mxIsSingle(prhs[2])) {
			const float* pfMatlabData = (const float *)mxGetData(prhs[2]);
			int i = 0;
			int col, row;
			for (col = 0; col < dims[1]; ++col) {
				for (row = 0; row < dims[0]; ++row) {
					pDataObject->getData2D()[row][col] = pfMatlabData[i];
					++i;
				}
			}
		} else {
			ASTRA_ASSERT(false);
		}
	}
}

//-----------------------------------------------------------------------------------------
/** geom = astra_mex_data2d('get_geometry', id);
 * 
 * Fetch the geometry of a 2d data object stored in the astra-library.
 * id: identifier of the 2d data object as stored in the astra-library.
 * geom: MATLAB-struct containing information about the used geometry.
 */
void astra_mex_data2d_get_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
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
	CFloat32Data2D* pDataObject = astra::CData2DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		if (pDataObject->getType() == CFloat32Data2D::PROJECTION) {
			CFloat32ProjectionData2D* pDataObject2 = dynamic_cast<CFloat32ProjectionData2D*>(pDataObject);
			plhs[0] = configToStruct(pDataObject2->getGeometry()->getConfiguration());
		}
		else if (pDataObject->getType() == CFloat32Data2D::VOLUME) {
			CFloat32VolumeData2D* pDataObject2 = dynamic_cast<CFloat32VolumeData2D*>(pDataObject);
			plhs[0] = configToStruct(pDataObject2->getGeometry()->getConfiguration());
		}
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_data2d('change_geometry', id, geom);
 *
 * Change the associated geometry of a 2d data object (volume or sinogram)
 * id: identifier of the 2d data object as stored in the astra-library.
 * geom: the new geometry struct, as created by astra_create_vol/proj_geom
 */
void astra_mex_data2d_change_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: check input
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value. \n");
		return;
	}

	// step2: get data object
	int iDataID = (int)(mxGetScalar(prhs[1]));
	CFloat32Data2D* pDataObject = astra::CData2DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	CFloat32ProjectionData2D* pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(pDataObject);

	if (pSinogram) {
		// Projection data

		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		Config* cfg = structToConfig("ProjectionGeometry2D", prhs[2]);
		// FIXME: Change how the base class is created. (This is duplicated
		// in 'create' and Projector2D.cpp.)
		std::string type = cfg->self.getAttribute("type");
		CProjectionGeometry2D* pGeometry;
		if (type == "sparse_matrix") {
			pGeometry = new CSparseMatrixProjectionGeometry2D();
		} else if (type == "fanflat") {
			//CFanFlatProjectionGeometry2D* pFanFlatProjectionGeometry = new CFanFlatProjectionGeometry2D();
			//pFanFlatProjectionGeometry->initialize(Config(node));
			//m_pProjectionGeometry = pFanFlatProjectionGeometry;
			pGeometry = new CFanFlatProjectionGeometry2D();	
		} else if (type == "fanflat_vec") {
			pGeometry = new CFanFlatVecProjectionGeometry2D();	
		} else {
			pGeometry = new CParallelProjectionGeometry2D();	
		}
		if (!pGeometry->initialize(*cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete pGeometry;
			delete cfg;
			return;
		}
		// If data is specified, check dimensions
		if (pGeometry->getDetectorCount() != pSinogram->getDetectorCount() || pGeometry->getProjectionAngleCount() != pSinogram->getAngleCount()) {
			mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
			delete pGeometry;
			delete cfg;
			return;
		}

		// If ok, change geometry
		pSinogram->changeGeometry(pGeometry);
		delete pGeometry;
		delete cfg;

		return;
	}

	CFloat32VolumeData2D* pVolume = dynamic_cast<CFloat32VolumeData2D*>(pDataObject);

	if (pVolume) {
		// Volume data

		// Read geometry
		if (!mxIsStruct(prhs[2])) {
			mexErrMsgTxt("Argument 3 is not a valid MATLAB struct.\n");
		}
		Config* cfg = structToConfig("VolumeGeometry2D", prhs[2]);
		CVolumeGeometry2D* pGeometry = new CVolumeGeometry2D();
		if (!pGeometry->initialize(*cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete cfg;
			delete pGeometry;
			return;
		}
		// If data is specified, check dimensions
		if (pGeometry->getGridColCount() != pVolume->getWidth() || pGeometry->getGridRowCount() != pVolume->getHeight()) {
			mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
			delete cfg;
			delete pGeometry;
			return;
		}

		// If ok, change geometry
		pVolume->changeGeometry(pGeometry);
		delete cfg;
		delete pGeometry;

	}

	mexErrMsgTxt("Data object not found or not initialized properly.\n");
	return;
}

//-----------------------------------------------------------------------------------------
/** data = astra_mex_data2d('get', id);
 *
 * Fetch data from the astra-library to a MATLAB matrix.
 * id: identifier of the 2d data object as stored in the astra-library.
 * data: MATLAB data
 */
void astra_mex_data2d_get(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: check input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value. \n");
		return;
	}

	// step2: get data object
	int iDataID = (int)(mxGetScalar(prhs[1]));
	CFloat32Data2D* pDataObject = astra::CData2DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleMatrix(pDataObject->getHeight(),	// # rows
									   pDataObject->getWidth(),		// # cols
									   mxREAL);						// datatype 64-bits
		double* out = mxGetPr(plhs[0]);
		int i = 0;
		int row, col;
		for (col = 0; col < pDataObject->getWidth(); ++col) {
			for (row = 0; row < pDataObject->getHeight(); ++row) {
				out[i] = pDataObject->getData2D()[row][col];
				++i;
			}
		}	
	}
	
}

//-----------------------------------------------------------------------------------------
/** data = astra_mex_data2d('get_single', id);
 *
 * Fetch data from the astra-library to a MATLAB matrix.
 * id: identifier of the 2d data object as stored in the astra-library.
 * data: MATLAB data
 */
void astra_mex_data2d_get_single(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: check input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value. \n");
		return;
	}

	// step2: get data object
	int iDataID = (int)(mxGetScalar(prhs[1]));
	CFloat32Data2D* pDataObject = astra::CData2DManager::getSingleton().get(iDataID);
	if (!pDataObject || !pDataObject->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		mwSize dims[2];
		dims[0] = pDataObject->getHeight();
		dims[1] = pDataObject->getWidth();
		plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
		float* out = (float *)mxGetData(plhs[0]);
		int i = 0;
		int row, col;
		for (col = 0; col < pDataObject->getWidth(); ++col) {
			for (row = 0; row < pDataObject->getHeight(); ++row) {
				out[i] = pDataObject->getData2D()[row][col];
				++i;
			}
		}	
	}
	
}

//-----------------------------------------------------------------------------------------
/** astra_mex_data2d('info');
 * 
 * Print information about all the 2d data objects currently stored in the astra-library.
 */
void astra_mex_data2d_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	mexPrintf("%s", astra::CData2DManager::getSingleton().info().c_str());
}

//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: get, get_single, delete, clear, set/store, create, get_geometry, change_geometry, info\n");
}

//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_data2d(type,...);
 */
void mexFunction(int nlhs, mxArray* plhs[],
				 int nrhs, const mxArray* prhs[])
{

	// INPUT0: Mode
	string sMode = "";
	if (1 <= nrhs) {
		sMode = mexToString(prhs[0]);	
	} else {
		printHelp();
		return;
	}

	initASTRAMex();

	// SWITCH (MODE)
	if (sMode ==  std::string("get")) { 
		astra_mex_data2d_get(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("get_single")) { 
		astra_mex_data2d_get_single(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("delete")) {	
		astra_mex_data2d_delete(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == "clear") {
		astra_mex_data2d_clear(nlhs, plhs, nrhs, prhs);
	} else if (sMode ==  std::string("store") ||
	           sMode ==  std::string("set")) {	
		astra_mex_data2d_store(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("create")) { 
		astra_mex_data2d_create(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("get_geometry")) { 
		astra_mex_data2d_get_geometry(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("change_geometry")) { 
		astra_mex_data2d_change_geometry(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("info")) { 
		astra_mex_data2d_info(nlhs, plhs, nrhs, prhs); 
	} else {
		printHelp();
	}

	return;
}


