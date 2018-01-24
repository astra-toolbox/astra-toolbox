/*
-----------------------------------------------------------------------
Copyright: 2010-2018, iMinds-Vision Lab, University of Antwerp
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

/** \file astra_mex_c.cpp
 *
 *  \brief Contains some basic "about" functions.
 */

#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include "astra/Globals.h"
#include "astra/AstraObjectManager.h"

#ifdef ASTRA_CUDA
#include "../cuda/2d/astra.h"
#include "../cuda/2d/util.h"
#include "astra/CompositeGeometryManager.h"
#endif


using namespace std;
using namespace astra;


//-----------------------------------------------------------------------------------------
/** astra_mex('credits');
 * 
 * Print Credits
 */
void astra_mex_credits(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	mexPrintf("The ASTRA Toolbox has been developed at the University of Antwerp and CWI, Amsterdam by\n");
	mexPrintf(" * Prof. dr. Joost Batenburg\n");
	mexPrintf(" * Prof. dr. Jan Sijbers\n");
	mexPrintf(" * Dr. Jeroen Bedorf\n");
	mexPrintf(" * Dr. Folkert Bleichrodt\n");
	mexPrintf(" * Dr. Andrei Dabravolski\n");
	mexPrintf(" * Dr. Willem Jan Palenstijn\n");
	mexPrintf(" * Dr. Tom Roelandts\n");
	mexPrintf(" * Dr. Wim van Aarle\n");
	mexPrintf(" * Dr. Gert Van Gompel\n");
	mexPrintf(" * Sander van der Maar, MSc.\n");
	mexPrintf(" * Gert Merckx, MSc.\n");
	mexPrintf(" * Daan Pelt, MSc.\n");
}

//-----------------------------------------------------------------------------------------
/** use_cuda = astra_mex('use_cuda');
 * 
 * Is CUDA enabled?
 */
void astra_mex_use_cuda(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(astra::cudaAvailable() ? 1 : 0);
	}
}

//-----------------------------------------------------------------------------------------
/** set_gpu_index = astra_mex('set_gpu_index');
 * 
 * Set active GPU
 */
void astra_mex_set_gpu_index(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
#ifdef ASTRA_CUDA
	bool usage = false;
	if (nrhs != 2 && nrhs != 4) {
		usage = true;
	}

	astra::SGPUParams params;
	params.memory = 0;

	if (!usage && nrhs >= 4) {
		std::string s = mexToString(prhs[2]);
		if (s != "memory") {
			usage = true;
		} else {
			params.memory = (size_t)mxGetScalar(prhs[3]);
		}
	}

	if (!usage && nrhs >= 2) {
		int n = mxGetN(prhs[1]) * mxGetM(prhs[1]);
		params.GPUIndices.resize(n);
		double* pdMatlabData = mxGetPr(prhs[1]);
		for (int i = 0; i < n; ++i)
			params.GPUIndices[i] = (int)pdMatlabData[i];


		astra::CCompositeGeometryManager::setGlobalGPUParams(params);


		// Set first GPU
		if (n >= 1) {
			bool ret = astraCUDA::setGPUIndex((int)pdMatlabData[0]);
			if (!ret)
				mexPrintf("Failed to set GPU %d\n", (int)pdMatlabData[0]);
		}
	}

	if (usage) {
		mexPrintf("Usage: astra_mex('set_gpu_index', index/indices [, 'memory', memory])");
	}
#endif
}

/** get_gpu_info = astra_mex('get_gpu_info');
 * 
 * Get GPU info
 */
void astra_mex_get_gpu_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
#ifdef ASTRA_CUDA
	int device = -1;
	if (nrhs >= 2 && mxIsDouble(prhs[1]) && mxGetN(prhs[1]) * mxGetM(prhs[1]) == 1 ) {
		device = (int)mxGetScalar(prhs[1]);
	}
	mexPrintf("%s\n", astraCUDA::getCudaDeviceString(device).c_str());
#endif
}


//-----------------------------------------------------------------------------------------
/** version_number = astra_mex('version');
 * 
 * Fetch the version number of the toolbox.
 */
void astra_mex_version(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(astra::getVersion());
	} else {
		mexPrintf("ASTRA Toolbox version %s\n", astra::getVersionString());
	}
}

//-----------------------------------------------------------------------------------------

void astra_mex_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 2) {
		mexErrMsgTxt("Usage: astra_mex('info', index/indices);\n");
		return;
	}

	for (int i = 1; i < nrhs; i++) {
		int iDataID = (int)(mxGetScalar(prhs[i]));
		CAstraObjectManagerBase *ptr;
		ptr = CAstraIndexManager::getSingleton().get(iDataID);
		if (ptr) {
			mexPrintf("%s\t%s\n", ptr->getType().c_str(), ptr->getInfo(iDataID).c_str());
		}
	}

}

//-----------------------------------------------------------------------------------------

void astra_mex_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 2) {
		mexErrMsgTxt("Usage: astra_mex('delete', index/indices);\n");
		return;
	}

	for (int i = 1; i < nrhs; i++) {
		int iDataID = (int)(mxGetScalar(prhs[i]));
		CAstraObjectManagerBase *ptr;
		ptr = CAstraIndexManager::getSingleton().get(iDataID);
		if (ptr)
			ptr->remove(iDataID);
	}

}



//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("   Valid modes: version, use_cuda, credits, set_gpu_index, info, delete\n");
}

//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex(type,...);
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
	if (sMode ==  std::string("version")) { 
		astra_mex_version(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("use_cuda")) {	
		astra_mex_use_cuda(nlhs, plhs, nrhs, prhs); 
	} else if (sMode ==  std::string("credits")) {	
		astra_mex_credits(nlhs, plhs, nrhs, prhs); 
	} else if (sMode == std::string("set_gpu_index")) {
		astra_mex_set_gpu_index(nlhs, plhs, nrhs, prhs);
	} else if (sMode == std::string("get_gpu_info")) {
		astra_mex_get_gpu_info(nlhs, plhs, nrhs, prhs);
	} else if (sMode == std::string("info")) {
		astra_mex_info(nlhs, plhs, nrhs, prhs);
	} else if (sMode == std::string("delete")) {
		astra_mex_delete(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}

	return;
}


