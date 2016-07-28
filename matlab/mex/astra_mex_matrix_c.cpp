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

/** \file astra_mex_matrix_c.cpp
 *
 *  \brief Create sparse (projection) matrices in the ASTRA workspace
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include <list>

#include "astra/Globals.h"

#include "astra/AstraObjectManager.h"

#include "astra/SparseMatrix.h"

using namespace std;
using namespace astra;

//-----------------------------------------------------------------------------------------
/** astra_mex_matrix('delete', id1, id2, ...);
 *
 * Delete one or more data objects currently stored in the astra-library.
 * id1, id2, ... : identifiers of the 2d data objects as stored in the astra-library.
 */
void astra_mex_matrix_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: read input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	// step2: delete all specified data objects
	for (int i = 1; i < nrhs; i++) {
		int iDataID = (int)(mxGetScalar(prhs[i]));
		CMatrixManager::getSingleton().remove(iDataID);
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_matrix('clear');
 *
 * Delete all data objects currently stored in the astra-library.
 */
void astra_mex_matrix_clear(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	CMatrixManager::getSingleton().clear();
}



static bool matlab_to_astra(const mxArray* _rhs, CSparseMatrix* _pMatrix)
{
	// Check input
	if (!mxIsSparse (_rhs)) {
		mexErrMsgTxt("Argument is not a valid MATLAB sparse matrix.\n");
		return false;
	}
	if (!_pMatrix->isInitialized()) {
		mexErrMsgTxt("Couldn't initialize data object.\n");
		return false;
	}

	unsigned int iHeight = mxGetM(_rhs);
	unsigned int iWidth = mxGetN(_rhs);
	unsigned long lSize = mxGetNzmax(_rhs);

	if (_pMatrix->m_lSize < lSize || _pMatrix->m_iHeight < iHeight) {
		// TODO: support resizing?
		mexErrMsgTxt("Matrix too large to store in this object.\n");
		return false;
	}

	// Transpose matrix, as matlab stores a matrix column-by-column
	// but we want it row-by-row.
	// 1. Compute sizes of rows. We store these in _pMatrix->m_plRowStarts.
	// 2. Fill data structure
	// Complexity: O( #rows + #entries )

	for (unsigned int i = 0; i <= iHeight; ++i)
		_pMatrix->m_plRowStarts[i] = 0;

	mwIndex *colStarts = mxGetJc(_rhs);
	mwIndex *rowIndices = mxGetIr(_rhs);
	double *floatValues = 0;
	mxLogical *boolValues = 0;
	bool bLogical = mxIsLogical(_rhs);
	if (bLogical)
		boolValues = mxGetLogicals(_rhs);
	else
		floatValues = mxGetPr(_rhs);

	for (mwIndex i = 0; i < colStarts[iWidth]; ++i) {
		unsigned int iRow = rowIndices[i];
		assert(iRow < iHeight);
		_pMatrix->m_plRowStarts[iRow+1]++;
	}

	// Now _pMatrix->m_plRowStarts[i+1] is the number of entries in row i

	for (unsigned int i = 1; i <= iHeight; ++i)
		_pMatrix->m_plRowStarts[i] += _pMatrix->m_plRowStarts[i-1];

	// Now _pMatrix->m_plRowStarts[i+1] is the number of entries in rows <= i,
	// so the intended start of row i+1

	int iCol = 0;
	for (mwIndex i = 0; i < colStarts[iWidth]; ++i) {
		while (i >= colStarts[iCol+1])
			iCol++;

		unsigned int iRow = rowIndices[i];
		assert(iRow < iHeight);
		float32 fVal;
		if (bLogical)
			fVal = (float32)boolValues[i];
		else
			fVal = (float32)floatValues[i];

		unsigned long lIndex = _pMatrix->m_plRowStarts[iRow]++;
		_pMatrix->m_pfValues[lIndex] = fVal;
		_pMatrix->m_piColIndices[lIndex] = iCol;
	}

	// Now _pMatrix->m_plRowStarts[i] is the start of row i+1

	for (unsigned int i = iHeight; i > 0; --i)
		_pMatrix->m_plRowStarts[i] = _pMatrix->m_plRowStarts[i-1];
	_pMatrix->m_plRowStarts[0] = 0;

#if 0
	// Debugging: dump matrix
	for (unsigned int i = 0; i < iHeight; ++i) {
		printf("Row %d: %ld-%ld\n", i, _pMatrix->m_plRowStarts[i], _pMatrix->m_plRowStarts[i+1]);
		for (unsigned long j = _pMatrix->m_plRowStarts[i]; j < _pMatrix->m_plRowStarts[i+1]; ++j) {
			printf("(%d,%d) = %f\n", i, _pMatrix->m_piColIndices[j], _pMatrix->m_pfValues[j]);
		}
	}
#endif

	return true;
}

static bool astra_to_matlab(const CSparseMatrix* _pMatrix, mxArray*& _lhs)
{
	if (!_pMatrix->isInitialized()) {
		mexErrMsgTxt("Uninitialized data object.\n");
		return false;
	}

	unsigned int iHeight = _pMatrix->m_iHeight;
	unsigned int iWidth = _pMatrix->m_iWidth;
	unsigned long lSize = _pMatrix->m_lSize;

	_lhs = mxCreateSparse(iHeight, iWidth, lSize, mxREAL);
	if (!mxIsSparse (_lhs)) {
		mexErrMsgTxt("Couldn't initialize matlab sparse matrix.\n");
		return false;
	}
	
	mwIndex *colStarts = mxGetJc(_lhs);
	mwIndex *rowIndices = mxGetIr(_lhs);
	double *floatValues = mxGetPr(_lhs);

	for (unsigned int i = 0; i <= iWidth; ++i)
		colStarts[i] = 0;

	for (unsigned int i = 0; i < _pMatrix->m_plRowStarts[iHeight]; ++i) {
		unsigned int iCol = _pMatrix->m_piColIndices[i];
		assert(iCol < iWidth);
		colStarts[iCol+1]++;
	}
	// Now _pMatrix->m_plRowStarts[i+1] is the number of entries in row i

	for (unsigned int i = 1; i <= iWidth; ++i)
		colStarts[i] += colStarts[i-1];
	// Now _pMatrix->m_plRowStarts[i+1] is the number of entries in rows <= i,
	// so the intended start of row i+1

	unsigned int iRow = 0;
	for (unsigned int i = 0; i < _pMatrix->m_plRowStarts[iHeight]; ++i) {
		while (i >= _pMatrix->m_plRowStarts[iRow+1])
			iRow++;

		unsigned int iCol = _pMatrix->m_piColIndices[i];
		assert(iCol < iWidth);
		double fVal = _pMatrix->m_pfValues[i];
		unsigned long lIndex = colStarts[iCol]++;
		floatValues[lIndex] = fVal;
		rowIndices[lIndex] = iRow;
	}
	// Now _pMatrix->m_plRowStarts[i] is the start of row i+1

	for (unsigned int i = iWidth; i > 0; --i)
		colStarts[i] = colStarts[i-1];
	colStarts[0] = 0;

	return true;
}

//-----------------------------------------------------------------------------------------
/** id = astra_mex_matrix('create', data);
 *   
 * Create a new matrix object in the astra-library.
 * data: a sparse MATLAB matrix containing the data.
 * id: identifier of the matrix object as it is now stored in the astra-library.
 */
void astra_mex_matrix_create(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{ 
	// step1: get datatype
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	if (!mxIsSparse (prhs[1])) {
		mexErrMsgTxt("Argument is not a valid MATLAB sparse matrix.\n");
		return;
	}

	unsigned int iHeight = mxGetM(prhs[1]);
	unsigned int iWidth = mxGetN(prhs[1]);
	unsigned long lSize = mxGetNzmax(prhs[1]);

	CSparseMatrix* pMatrix = new CSparseMatrix(iHeight, iWidth, lSize);

	// Check initialization
	if (!pMatrix->isInitialized()) {
		mexErrMsgTxt("Couldn't initialize data object.\n");
		delete pMatrix;
		return;
	}

	bool bResult = matlab_to_astra(prhs[1], pMatrix);

	if (!bResult) {
		mexErrMsgTxt("Failed to create data object.\n");
		delete pMatrix;
		return;
	}

	// store data object
	int iIndex = CMatrixManager::getSingleton().store(pMatrix);

	// return data id
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(iIndex);
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_matrix('store', id, data);
 *
 * Store a sparse MATLAB matrix in an existing astra matrix dataobject. 
 * id: identifier of the 2d data object as stored in the astra-library.
 * data: a sparse MATLAB matrix.
 */
void astra_mex_matrix_store(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
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

	// step2: get data object
	CSparseMatrix* pMatrix = astra::CMatrixManager::getSingleton().get(iDataID);
	if (!pMatrix || !pMatrix->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	bool bResult = matlab_to_astra(prhs[2], pMatrix);
	if (!bResult) {
		mexErrMsgTxt("Failed to store matrix.\n");
	}
}

//-----------------------------------------------------------------------------------------
/** geom = astra_mex_matrix('get_size', id);
 * 
 * Fetch the dimensions and size of a matrix stored in the astra-library.
 * id: identifier of the 2d data object as stored in the astra-library.
 * geom: a 1x2 matrix containing [rows, columns]
 */
void astra_mex_matrix_get_size(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	if (!mxIsDouble(prhs[1])) {
		mexErrMsgTxt("Identifier should be a scalar value. \n");
		return;
	}
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CSparseMatrix* pMatrix = astra::CMatrixManager::getSingleton().get(iDataID);
	if (!pMatrix || !pMatrix->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	// TODO
}

//-----------------------------------------------------------------------------------------
/** data = astra_mex_matrix('get', id);
 *
 * Fetch data from the astra-library to a MATLAB matrix.
 * id: identifier of the matrix data object as stored in the astra-library.
 * data: MATLAB
 */
void astra_mex_matrix_get(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
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
	int iDataID = (int)(mxGetScalar(prhs[1]));

	// step2: get data object
	CSparseMatrix* pMatrix = astra::CMatrixManager::getSingleton().get(iDataID);
	if (!pMatrix || !pMatrix->isInitialized()) {
		mexErrMsgTxt("Data object not found or not initialized properly.\n");
		return;
	}

	// create output
	if (1 <= nlhs) {
		bool bResult = astra_to_matlab(pMatrix, plhs[0]);
		if (!bResult) {
			mexErrMsgTxt("Failed to get matrix.\n");
		}
	}
	
}

//-----------------------------------------------------------------------------------------
/** astra_mex_matrix('info');
 * 
 * Print information about all the matrix objects currently stored in the astra-library.
 */
void astra_mex_matrix_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	mexPrintf("%s", astra::CMatrixManager::getSingleton().info().c_str());
}

//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: get, delete, clear, store, create, get_size, info\n");
}

//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_matrix(type,...);
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
		astra_mex_matrix_get(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("delete")) {	
		astra_mex_matrix_delete(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == "clear") {
		astra_mex_matrix_clear(nlhs, plhs, nrhs, prhs);
	} else if (sMode ==  std::string("store")) {	
		astra_mex_matrix_store(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("create")) { 
		astra_mex_matrix_create(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("get_size")) { 
		astra_mex_matrix_get_size(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("info")) { 
		astra_mex_matrix_info(nlhs, plhs, nrhs, prhs); 
	} else {
		printHelp();
	}

	return;
}


