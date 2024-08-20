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

/** \file astra_mex_algorithm_c.cpp
 *
 *  \brief Creates and manages algorithms (reconstruction,projection,...).
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"
#include "astra/Globals.h"

#ifdef USE_MATLAB_UNDOCUMENTED
extern "C" { bool utIsInterruptPending(); }
#endif




#include "astra/AstraObjectManager.h"
#include "astra/AstraObjectFactory.h"

#include "astra/XMLNode.h"
#include "astra/XMLDocument.h"

using namespace std;
using namespace astra;
//-----------------------------------------------------------------------------------------
/** id = astra_mex_algorithm('create', cfg);
 *
 * Create and configure a new algorithm object.
 * cfg: MATLAB struct containing the configuration parameters, see doxygen documentation for details.
 * id: identifier of the algorithm object as it is now stored in the astra-library.
 */
void astra_mex_algorithm_create(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}

	if (!mxIsStruct(prhs[1])) {
		mexErrMsgTxt("Argument 1 not a valid MATLAB struct.");
	}

	// turn MATLAB struct to an XML-based Config object
	Config* cfg = structToConfig("Algorithm", prhs[1]);

	CAlgorithm* pAlg = CAlgorithmFactory::getSingleton().create(cfg->self.getAttribute("type"));
	if (!pAlg) {
		delete cfg;
		mexErrMsgTxt("Unknown algorithm type.");
	}

	// create algorithm
	if (!pAlg->initialize(*cfg)) {
		delete cfg;
		delete pAlg;
		mexErrMsgWithAstraLog("Unable to initialize algorithm.");
	}
	delete cfg;

	// store algorithm
	int iIndex = CAlgorithmManager::getSingleton().store(pAlg);

	// step4: set output
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(iIndex);
	}

}

#ifdef USE_MATLAB_UNDOCUMENTED
bool checkMatlabInterrupt() {
	return utIsInterruptPending();
}
#endif

//-----------------------------------------------------------------------------------------
/** astra_mex_algorithm('run', id); or astra_mex_algorithm('iterate', id, iterations);
 *
 * Run or do iterations on a certain algorithm.
 * id: identifier of the algorithm object as stored in the astra-library.
 * iterations: if the algorithm is iterative, this specifies the number of iterations to perform.
 */
void astra_mex_algorithm_run(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: get input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}
	int iAid = (int)(mxGetScalar(prhs[1]));
	int iIterations = 0;
	if (3 <= nrhs) {
		iIterations = (int)(mxGetScalar(prhs[2]));
	}

	// step2: get algorithm object
	CAlgorithm* pAlg = CAlgorithmManager::getSingleton().get(iAid);
	if (!pAlg) {
		mexErrMsgTxt("Invalid algorithm ID.");
	}
	if (!pAlg->isInitialized()) {
		mexErrMsgTxt("Algorithm ID exists but is not initialized.");
	}

	// step3: perform actions

#ifdef USE_MATLAB_UNDOCUMENTED
	setShouldAbortHook(&checkMatlabInterrupt);
#endif

	pAlg->run(iIterations);
}
//-----------------------------------------------------------------------------------------
/** astra_mex_algorithm('get_res_norm', id);
 *
 * Get the L2-norm of the residual sinogram. Not all algorithms
 * support this operation.
 */
void astra_mex_algorithm_get_res_norm(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: get input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}
	int iAid = (int)(mxGetScalar(prhs[1]));

	// step2: get algorithm object
	CAlgorithm* pAlg = CAlgorithmManager::getSingleton().get(iAid);
	if (!pAlg) {
		mexErrMsgTxt("Invalid algorithm ID.");
	}
	if (!pAlg->isInitialized()) {
		mexErrMsgTxt("Algorithm ID exists but is not initialized.");
	}

	CReconstructionAlgorithm2D* pAlg2D = dynamic_cast<CReconstructionAlgorithm2D*>(pAlg);
	CReconstructionAlgorithm3D* pAlg3D = dynamic_cast<CReconstructionAlgorithm3D*>(pAlg);

	float res = 0.0f;
	bool ok;
	if (pAlg2D)
		ok = pAlg2D->getResidualNorm(res);
	else if (pAlg3D)
		ok = pAlg3D->getResidualNorm(res);
	else
		ok = false;

	if (!ok) {
		mexErrMsgTxt("Operation not supported.");
	}

	plhs[0] = mxCreateDoubleScalar(res);
}

//-----------------------------------------------------------------------------------------
/** astra_mex_algorithm('delete', id1, id2, ...);
 *
 * Delete one or more algorithm objects currently stored in the astra-library. 
 * id1, id2, ... : identifiers of the algorithm objects as stored in the astra-library.
 */
void astra_mex_algorithm_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: get algorithm ID
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments. See the help document for a detailed argument list.");
	}

	for (int i = 1; i < nrhs; i++) {
		int iAid = (int)(mxGetScalar(prhs[i]));
		CAlgorithmManager::getSingleton().remove(iAid);
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_algorithm('clear');
 *
 * Delete all algorithm objects currently stored in the astra-library.
 */
void astra_mex_algorithm_clear(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	CAlgorithmManager::getSingleton().clear();
}

//-----------------------------------------------------------------------------------------
/** astra_mex_algorithm('info');
 *
 * Print information about all the algorithm objects currently stored in the astra-library.
 */
void astra_mex_algorithm_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	mexPrintf("%s", astra::CAlgorithmManager::getSingleton().info().c_str());
}

//-----------------------------------------------------------------------------------------
static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: create, info, delete, clear, run/iterate, get_res_norm\n");
}

//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_algorithm(mode, ...);
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

	// SWITCH (MODE)
	if (sMode == "create") {
		astra_mex_algorithm_create(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "info") {
		astra_mex_algorithm_info(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "delete") {
		astra_mex_algorithm_delete(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "clear") {
		astra_mex_algorithm_clear(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "run" || sMode == "iterate") {
		astra_mex_algorithm_run(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "get_res_norm") {
		astra_mex_algorithm_get_res_norm(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}
	return;
}
