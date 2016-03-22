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

#include "astra/CompositeGeometryManager.h"

using namespace std;
using namespace astra;

#define USE_MATLAB_UNDOCUMENTED


class CFloat32CustomMemory_simple : public astra::CFloat32CustomMemory {
public:
	CFloat32CustomMemory_simple(float *ptr) { m_fPtr = ptr; }
	~CFloat32CustomMemory_simple() { }
};

#ifdef ASTRA_CUDA

template<typename Type>
void clean_vector(vector<Type *> & vec, const int limit)
{
	for (mwIndex itClean = 0; itClean < limit; itClean++) {
		delete vec[itClean];
	}
}

bool is_cuda(vector<astra::CProjector3D *> & pProjectors)
{
	// I know the logic is counter intuitive here, but it uses an implicit
	// conversion from pointer to boolean of the operator !
	const mwSize num_projectors = pProjectors.size();
	bool is_not_cuda = false;
	for (mwIndex itProj = 0; itProj < num_projectors; itProj++)
	{
		is_not_cuda |= !(dynamic_cast<CCudaProjector3D*>(pProjectors[itProj]));
	}
	return !is_not_cuda;
}

bool get_projectors(const mxArray * const projs,
                    vector<astra::CProjector3D *> & pProjectors,
                    vector<astra::CVolumeGeometry3D *> & pVolGeoms,
                    vector<astra::CProjectionGeometry3D *> & pProjGeoms)
{
	const double * const dPids = (const double *)mxGetData(projs);
	const mwSize num_projectors = pProjectors.size();
	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		int iPid = (int)(dPids[itPid]);
		pProjectors[itPid] = astra::CProjector3DManager::getSingleton().get(iPid);
		if (!pProjectors[itPid]) {
			mexErrMsgTxt("One of the projectors was not found.");
			return false;
		}
		if (!pProjectors[itPid]->isInitialized()) {
			mexErrMsgTxt("One of the projectors was not initialized.");
			return false;
		}
		pVolGeoms[itPid] = pProjectors[itPid]->getVolumeGeometry();
		pProjGeoms[itPid] = pProjectors[itPid]->getProjectionGeometry();
	}
	return true;
}

template<typename DataType, typename GeometryType>
DataType * load_data(const mxArray * const data, GeometryType * const pGeom)
{
	if (!checkDataType(data)) {
		mexErrMsgTxt("Data must be single or double.");
		return 0;
	}

	if (!checkDataSize(data, pGeom)) {
		mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
		return 0;
	}

	// Allocate input data
	DataType * pInput;
	if (mxIsSingle(data)) {
		astra::CFloat32CustomMemory * m = new CFloat32CustomMemory_simple((float *)mxGetData(data));
		pInput = new DataType(pGeom, m);
	} else {
		pInput = new DataType(pGeom);
		copyMexToCFloat32Array(data, pInput->getData(), pInput->getSize());
	}
	return pInput;
}

template<typename DataType>
mxArray * produce_output(vector<mxArray *> & pOutputMxs,
                         const vector<DataType *> & pOutput,
                         const vector<const mxArray * > & data)
{
	mxArray * pOutputMx;
	const mwSize num_projectors = data.size();

	if (num_projectors > 1)	{
		pOutputMx = mxCreateCellMatrix(num_projectors, 1);

		for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
			if (mxIsSingle(data[itPid])) {

			} else {
				pOutputMxs[itPid] = createEquivMexArray<mxDOUBLE_CLASS>(pOutput[itPid]);
				copyCFloat32ArrayToMex(pOutput[itPid]->getData(), pOutputMxs[itPid]);
			}
			mxSetCell(pOutputMx, itPid, pOutputMxs[itPid]);
		}
	} else if (mxIsSingle(data[0])) {
		pOutputMx = pOutputMxs[0];
	} else {
		pOutputMx = createEquivMexArray<mxDOUBLE_CLASS>(pOutput[0]);
		copyCFloat32ArrayToMex(pOutput[0]->getData(), pOutputMx);
	}

	return pOutputMx;
}

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

	mwSize num_projectors = mxGetNumberOfElements(prhs[1]);
	if (!num_projectors) {
		mexErrMsgTxt("No projectors passed!");
		return;
	}
	vector<astra::CProjector3D *> pProjectors(num_projectors);
	vector<astra::CVolumeGeometry3D *> pVolGeoms(num_projectors);
	vector<astra::CProjectionGeometry3D *> pProjGeoms(num_projectors);

	if (!get_projectors(prhs[1], pProjectors, pVolGeoms, pProjGeoms)) {
		return;
	}

	if (!is_cuda(pProjectors)) {
		mexErrMsgTxt("Only CUDA projectors are currently supported.");
		return;
	}

	vector<const mxArray *> data(num_projectors);

	if (num_projectors > 1) {
		if (!mxIsCell(prhs[2]) || (mxGetNumberOfElements(prhs[2]) != num_projectors)) {
			mexErrMsgTxt("When using multiple projectors, a cell array with an equal number of volumes should be passed");
			return;
		}
		for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
			data[itPid] = mxGetCell(prhs[2], itPid);
		}
	} else {
		data[0] = prhs[2];
	}

	vector<astra::CFloat32VolumeData3DMemory *> pInput(num_projectors);

	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		pInput[itPid] = load_data<astra::CFloat32VolumeData3DMemory>(data[itPid], pVolGeoms[itPid]);
		if (!pInput[itPid]) {
			// Cleaning
			clean_vector(pInput, itPid);
			return;
		}
	}

	// Allocate output data
	// If the input is single, we also allocate single output.
	// Otherwise, double.
	vector<astra::CFloat32ProjectionData3DMemory *> pOutput(num_projectors);
	vector<mxArray *> pOutputMxs(num_projectors);

	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		if (mxIsSingle(data[itPid])) {
			mwSize dims[3];
			dims[0] = pProjGeoms[itPid]->getDetectorColCount();
			dims[1] = pProjGeoms[itPid]->getProjectionCount();
			dims[2] = pProjGeoms[itPid]->getDetectorRowCount();

			// Allocate uninitialized mxArray of size dims.
			// (It will be zeroed by CudaForwardProjectionAlgorithm3D)
			const mwSize zero_dims[2] = {0, 0};
			pOutputMxs[itPid] = mxCreateNumericArray(2, zero_dims, mxSINGLE_CLASS, mxREAL);
			mxSetDimensions(pOutputMxs[itPid], dims, 3);
			const mwSize num_elems = mxGetNumberOfElements(pOutputMxs[itPid]);
			const mwSize elem_size = mxGetElementSize(pOutputMxs[itPid]);
			mxSetData(pOutputMxs[itPid], mxMalloc(elem_size * num_elems));

			astra::CFloat32CustomMemory* m = new CFloat32CustomMemory_simple((float *)mxGetData(pOutputMxs[itPid]));
			pOutput[itPid] = new astra::CFloat32ProjectionData3DMemory(pProjGeoms[itPid], m);
		} else {
			pOutput[itPid] = new astra::CFloat32ProjectionData3DMemory(pProjGeoms[itPid]);
		}
	}

	// Perform FP

	CCompositeGeometryManager cgm;

	CCompositeGeometryManager::TJobList L;
	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		L.push_back(cgm.createJobFP(pProjectors[itPid], pInput[itPid], pOutput[itPid]));
	}
	cgm.doJobs(L);

	plhs[0] = produce_output(pOutputMxs, pOutput, data);

	clean_vector(pOutput, num_projectors);
	clean_vector(pInput, num_projectors);
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

	mwSize num_projectors = mxGetNumberOfElements(prhs[1]);
	if (!num_projectors) {
		mexErrMsgTxt("No projectors passed!");
		return;
	}
	vector<astra::CProjector3D *> pProjectors(num_projectors);
	vector<astra::CVolumeGeometry3D *> pVolGeoms(num_projectors);
	vector<astra::CProjectionGeometry3D *> pProjGeoms(num_projectors);

	if (!get_projectors(prhs[1], pProjectors, pVolGeoms, pProjGeoms)) {
		return;
	}

	if (!is_cuda(pProjectors)) {
		mexErrMsgTxt("Only CUDA projectors are currently supported.");
		return;
	}

	vector<const mxArray *> data(num_projectors);

	if (num_projectors > 1) {
		if (!mxIsCell(prhs[2]) || (mxGetNumberOfElements(prhs[2]) != num_projectors)) {
			mexErrMsgTxt("When using multiple projectors, a cell array with an equal number of projection data stacks should be passed");
			return;
		}
		for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
			data[itPid] = mxGetCell(prhs[2], itPid);
		}
	} else {
		data[0] = prhs[2];
	}

	// Allocate input data
	vector<astra::CFloat32ProjectionData3DMemory *> pInput(num_projectors);

	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		pInput[itPid] = load_data<astra::CFloat32ProjectionData3DMemory>(data[itPid], pProjGeoms[itPid]);
		if (!pInput[itPid]) {
			// Cleaning
			clean_vector(pInput, itPid);
			return;
		}
	}

	// Allocate output data
	// If the input is single, we also allocate single output.
	// Otherwise, double.
	vector<astra::CFloat32VolumeData3DMemory *> pOutput(num_projectors);
	vector<mxArray *> pOutputMxs(num_projectors);

	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		if (mxIsSingle(data[itPid])) {
			mwSize dims[3];
			dims[0] = pVolGeoms[itPid]->getGridColCount();
			dims[1] = pVolGeoms[itPid]->getGridRowCount();
			dims[2] = pVolGeoms[itPid]->getGridSliceCount();

			// Allocate uninitialized mxArray of size dims.
			// (It will be zeroed by CudaForwardProjectionAlgorithm3D)
			const mwSize zero_dims[2] = {0, 0};
			pOutputMxs[itPid] = mxCreateNumericArray(2, zero_dims, mxSINGLE_CLASS, mxREAL);
			mxSetDimensions(pOutputMxs[itPid], dims, 3);
			const mwSize num_elems = mxGetNumberOfElements(pOutputMxs[itPid]);
			const mwSize elem_size = mxGetElementSize(pOutputMxs[itPid]);
			mxSetData(pOutputMxs[itPid], mxMalloc(elem_size * num_elems));

			astra::CFloat32CustomMemory* m = new CFloat32CustomMemory_simple((float *)mxGetData(pOutputMxs[itPid]));
			pOutput[itPid] = new astra::CFloat32VolumeData3DMemory(pVolGeoms[itPid], m);
		} else {
			pOutput[itPid] = new astra::CFloat32VolumeData3DMemory(pVolGeoms[itPid]);
		}
	}

	// Perform BP

	CCompositeGeometryManager cgm;

	CCompositeGeometryManager::TJobList L;
	for (mwIndex itPid = 0; itPid < num_projectors; itPid++) {
		L.push_back(cgm.createJobBP(pProjectors[itPid], pOutput[itPid], pInput[itPid]));
	}
	cgm.doJobs(L);

	plhs[0] = produce_output(pOutputMxs, pOutput, data);

	clean_vector(pOutput, num_projectors);
	clean_vector(pInput, num_projectors);
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


