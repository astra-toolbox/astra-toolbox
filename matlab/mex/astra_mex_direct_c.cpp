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

/** \file astra_mex_direct_c.cpp
 *
 *  \brief Utility functions for low-overhead FP and BP calls.
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
#include "astra/CudaProjector3D.h"
#include "astra/Projector3D.h"
#include "astra/Float32ProjectionData3DMemory.h"
#include "astra/Float32VolumeData3DMemory.h"

#include "astra/CudaForwardProjectionAlgorithm3D.h"

#include "astra/CudaBackProjectionAlgorithm3D.h"

using namespace std;
using namespace astra;

#define USE_MATLAB_UNDOCUMENTED


class CFloat32CustomMemory_simple : public astra::CFloat32CustomMemory {
public:
	CFloat32CustomMemory_simple(float *ptr) { m_fPtr = ptr; }
	~CFloat32CustomMemory_simple() { }
};

#ifdef ASTRA_CUDA

//-----------------------------------------------------------------------------------------
/**
 * projection = astra_mex_direct_c('FP3D', projector_id, volume);
 * Both 'projection' and 'volume' are Matlab arrays.
 */
void astra_mex_direct_fp3d(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{
	// TODO: Add an optional way of specifying extra options

	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments. Syntax: astra_mex_direct_c('FP3D', projector_id, data)");
		return;
	}

	int iPid = (int)(mxGetScalar(prhs[1]));
	astra::CProjector3D* pProjector;
	pProjector = astra::CProjector3DManager::getSingleton().get(iPid);
	if (!pProjector) {
		mexErrMsgTxt("Projector not found.");
		return;
	}
	if (!pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.");
		return;
	}
	bool isCuda = false;
	if (dynamic_cast<CCudaProjector3D*>(pProjector))
		isCuda = true;
	if (!isCuda) {
		mexErrMsgTxt("Only CUDA projectors are currently supported.");
		return;
	}

	astra::CVolumeGeometry3D* pVolGeom = pProjector->getVolumeGeometry();
	astra::CProjectionGeometry3D* pProjGeom = pProjector->getProjectionGeometry();

	const mxArray* const data = prhs[2];
	if (!checkDataType(data)) {
		mexErrMsgTxt("Data must be single or double.");
		return;
	}

	if (!checkDataSize(data, pVolGeom)) {
		mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
		return;
	}


	// Allocate input data
	astra::CFloat32VolumeData3DMemory* pInput;
	if (mxIsSingle(data)) {
		astra::CFloat32CustomMemory* m = new CFloat32CustomMemory_simple((float *)mxGetData(data));
		pInput = new astra::CFloat32VolumeData3DMemory(pVolGeom, m);
	} else {
		pInput = new astra::CFloat32VolumeData3DMemory(pVolGeom);
		copyMexToCFloat32Array(data, pInput->getData(), pInput->getSize());
	}


	// Allocate output data
	// If the input is single, we also allocate single output.
	// Otherwise, double.
	astra::CFloat32ProjectionData3DMemory* pOutput;
	mxArray *pOutputMx;
	if (mxIsSingle(data)) {
		mwSize dims[3];
		dims[0] = pProjGeom->getDetectorColCount();
		dims[1] = pProjGeom->getProjectionCount();
		dims[2] = pProjGeom->getDetectorRowCount();

		// Allocate uninitialized mxArray of size dims.
		// (It will be zeroed by CudaForwardProjectionAlgorithm3D)
		const mwSize zero_dims[2] = {0, 0};
		pOutputMx = mxCreateNumericArray(2, zero_dims, mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(pOutputMx, dims, 3);
		const mwSize num_elems = mxGetNumberOfElements(pOutputMx);
		const mwSize elem_size = mxGetElementSize(pOutputMx);
		mxSetData(pOutputMx, mxMalloc(elem_size * num_elems));

		astra::CFloat32CustomMemory* m = new CFloat32CustomMemory_simple((float *)mxGetData(pOutputMx));
		pOutput = new astra::CFloat32ProjectionData3DMemory(pProjGeom, m);
	} else {
		pOutput = new astra::CFloat32ProjectionData3DMemory(pProjGeom);
	}

	// Perform FP

	astra::CCudaForwardProjectionAlgorithm3D* pAlg;
	pAlg = new astra::CCudaForwardProjectionAlgorithm3D();
	pAlg->initialize(pProjector, pOutput, pInput);

	if (!pAlg->isInitialized()) {
		mexErrMsgTxt("Error initializing algorithm.");
		// TODO: Delete pOutputMx?
		delete pAlg;
		delete pInput;
		delete pOutput;
		return;
	}

	pAlg->run();

	delete pAlg;


	if (mxIsSingle(data)) {

	} else {
		pOutputMx = createEquivMexArray<mxDOUBLE_CLASS>(pOutput);
		copyCFloat32ArrayToMex(pOutput->getData(), pOutputMx);
	}
	plhs[0] = pOutputMx;

	delete pOutput;
	delete pInput;
}
//-----------------------------------------------------------------------------------------
/**
 * projection = astra_mex_direct_c('BP3D', projector_id, volume);
 * Both 'projection' and 'volume' are Matlab arrays.
 */
void astra_mex_direct_bp3d(int& nlhs, mxArray* plhs[], int& nrhs, const mxArray* prhs[])
{
	// TODO: Add an optional way of specifying extra options

	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments. Syntax: astra_mex_direct_c('BP3D', projector_id, data)");
		return;
	}

	int iPid = (int)(mxGetScalar(prhs[1]));
	astra::CProjector3D* pProjector;
	pProjector = astra::CProjector3DManager::getSingleton().get(iPid);
	if (!pProjector) {
		mexErrMsgTxt("Projector not found.");
		return;
	}
	if (!pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.");
		return;
	}
	bool isCuda = false;
	if (dynamic_cast<CCudaProjector3D*>(pProjector))
		isCuda = true;
	if (!isCuda) {
		mexErrMsgTxt("Only CUDA projectors are currently supported.");
		return;
	}

	astra::CVolumeGeometry3D* pVolGeom = pProjector->getVolumeGeometry();
	astra::CProjectionGeometry3D* pProjGeom = pProjector->getProjectionGeometry();

	const mxArray* const data = prhs[2];
	if (!checkDataType(data)) {
		mexErrMsgTxt("Data must be single or double.");
		return;
	}

	if (!checkDataSize(data, pProjGeom)) {
		mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
		return;
	}


	// Allocate input data
	astra::CFloat32ProjectionData3DMemory* pInput;
	if (mxIsSingle(data)) {
		astra::CFloat32CustomMemory* m = new CFloat32CustomMemory_simple((float *)mxGetData(data));
		pInput = new astra::CFloat32ProjectionData3DMemory(pProjGeom, m);
	} else {
		pInput = new astra::CFloat32ProjectionData3DMemory(pProjGeom);
		copyMexToCFloat32Array(data, pInput->getData(), pInput->getSize());
	}


	// Allocate output data
	// If the input is single, we also allocate single output.
	// Otherwise, double.
	astra::CFloat32VolumeData3DMemory* pOutput;
	mxArray *pOutputMx;
	if (mxIsSingle(data)) {
		mwSize dims[3];
		dims[0] = pVolGeom->getGridColCount();
		dims[1] = pVolGeom->getGridRowCount();
		dims[2] = pVolGeom->getGridSliceCount();

		// Allocate uninitialized mxArray of size dims.
		// (It will be zeroed by CudaBackProjectionAlgorithm3D)
		const mwSize zero_dims[2] = {0, 0};
		pOutputMx = mxCreateNumericArray(2, zero_dims, mxSINGLE_CLASS, mxREAL);
		mxSetDimensions(pOutputMx, dims, 3);
		const mwSize num_elems = mxGetNumberOfElements(pOutputMx);
		const mwSize elem_size = mxGetElementSize(pOutputMx);
		mxSetData(pOutputMx, mxMalloc(elem_size * num_elems));

		astra::CFloat32CustomMemory* m = new CFloat32CustomMemory_simple((float *)mxGetData(pOutputMx));
		pOutput = new astra::CFloat32VolumeData3DMemory(pVolGeom, m);
	} else {
		pOutput = new astra::CFloat32VolumeData3DMemory(pVolGeom);
	}

	// Perform BP

	astra::CCudaBackProjectionAlgorithm3D* pAlg;
	pAlg = new astra::CCudaBackProjectionAlgorithm3D();
	pAlg->initialize(pProjector, pInput, pOutput);

	if (!pAlg->isInitialized()) {
		mexErrMsgTxt("Error initializing algorithm.");
		// TODO: Delete pOutputMx?
		delete pAlg;
		delete pInput;
		delete pOutput;
		return;
	}

	pAlg->run();

	delete pAlg;


	if (mxIsSingle(data)) {

	} else {
		pOutputMx = createEquivMexArray<mxDOUBLE_CLASS>(pOutput);
		copyCFloat32ArrayToMex(pOutput->getData(), pOutputMx);
	}
	plhs[0] = pOutputMx;

	delete pOutput;
	delete pInput;
}

#endif

//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: FP3D, BP3D\n");
}


//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_direct_c(mode,...);
 */
void mexFunction(int nlhs, mxArray* plhs[],
				 int nrhs, const mxArray* prhs[])
{

	// INPUT: Mode
	string sMode;
	if (1 <= nrhs) {
		sMode = mexToString(prhs[0]);
	} else {
		printHelp();
		return;
	}

#ifndef ASTRA_CUDA
	mexErrMsgTxt("Only CUDA projectors are currently supported.");
	return;
#else

	// 3D data
	if (sMode == "FP3D") {
		astra_mex_direct_fp3d(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "BP3D") {
		astra_mex_direct_bp3d(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}
#endif

	return;
}


