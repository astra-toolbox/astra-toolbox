/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
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

/** \file astra_mex_projector_c.cpp
 *
 *  \brief Create and manage 2d projectors in the ASTRA workspace
 */
#include "astra/Globals.h"

#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include "astra/AstraObjectManager.h"
#include "astra/Projector2D.h"
#include "astra/AstraObjectFactory.h"

#include "astra/Float32VolumeData2D.h"

#include "astra/ProjectionGeometry2D.h"
#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/VolumeGeometry2D.h"


#include <map>
#include <vector>

using namespace std;
using namespace astra;

//-----------------------------------------------------------------------------------------
/** id = astra_mex_projector('create', cfg);
 *
 * Create and configure a new projector object.
 * cfg: MATLAB struct containing the configuration parameters, see doxygen documentation for details.
 * id: identifier of the projector object as it is now stored in the astra-library.
 */
void astra_mex_projector_create(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	int iIndex = 0;

	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	if (!mxIsStruct(prhs[1])) {
		mexErrMsgTxt("Argument 1 not a valid MATLAB struct. \n");
	}


	// turn MATLAB struct to an XML-based Config object
	Config* cfg = structToConfig("Projector2D", prhs[1]);

	// create algorithm
	CProjector2D* pProj = CProjector2DFactory::getSingleton().create(cfg->self.getAttribute("type"));
	if (pProj == NULL) {
		delete cfg;
		mexErrMsgTxt("Unknown Projector2D. \n");
		return;
	}

	// create algorithm
	if (!pProj->initialize(*cfg)) {
		delete cfg;
		delete pProj;
		mexErrMsgTxt("Unable to initialize Projector2D. \n");
		return;
	}
	delete cfg;

	// store projector
	iIndex = CProjector2DManager::getSingleton().store(pProj);

	// step4: set output
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(iIndex);
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_projector('delete', id1, id2, ...);
 *
 * Delete one or more projector objects currently stored in the astra-library. 
 * id1, id2, ... : identifiers of the projector objects as stored in the astra-library.
 */
void astra_mex_projector_delete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: read input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}

	for (int i = 1; i < nrhs; i++) {
		int iPid = (int)(mxGetScalar(prhs[i]));
		CProjector2DManager::getSingleton().remove(iPid);
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_projector('clear');
 *
 * Delete all projector objects currently stored in the astra-library.
 */
void astra_mex_projector_clear(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	CProjector2DManager::getSingleton().clear();
}

//-----------------------------------------------------------------------------------------
/** astra_mex_projector('info');
 *
 * Print information about all the projector objects currently stored in the astra-library.
 */
void astra_mex_projector_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	mexPrintf("%s", astra::CProjector2DManager::getSingleton().info().c_str());
}

//-----------------------------------------------------------------------------------------
/** proj_geom = astra_mex_projector('projection_geometry', id);
 *
 * Fetch the projection geometry of a certain projector.
 * id: identifier of the projector object as stored in the astra-library.
 * proj_geom: MATLAB struct containing all information about the projection geometry
*/
void astra_mex_projector_projection_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: read input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iPid = (int)(mxGetScalar(prhs[1]));

	// step2: get projector
	CProjector2D* pProjector = CProjector2DManager::getSingleton().get(iPid);
	if (!pProjector || !pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.\n");
		return;
	}

	// step3: get projection_geometry and turn it into a MATLAB struct
	if (1 <= nlhs) {
		Config *cfg =  pProjector->getProjectionGeometry()->getConfiguration();
		plhs[0] = configToStruct(cfg);
		delete cfg;
	}
}

//-----------------------------------------------------------------------------------------
/** vol_geom = astra_mex_projector('volume_geometry', id);
 *
 * Fetch the volume geometry of a certain projector.
 * id: identifier of the projector object as stored in the astra-library.
 * vol_geom: MATLAB struct containing all information about the volume geometry
 */
void astra_mex_projector_volume_geometry(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	// step1: read input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iPid = (int)(mxGetScalar(prhs[1]));

	// step2: get projector
	CProjector2D* pProjector = CProjector2DManager::getSingleton().get(iPid);
	if (!pProjector || !pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.\n");
		return;
	}

	// step3: get projection_geometry and turn it into a MATLAB struct
	if (1 <= nlhs) {
		Config *cfg = pProjector->getVolumeGeometry()->getConfiguration();
		plhs[0] = configToStruct(cfg);
		delete cfg;
	}
}

//-----------------------------------------------------------------------------------------
/** weights = astra_mex_projector('weights_single_ray', id, projection_index, detector_index);
 *
 * Calculate the nonzero weights of a certain projection ray.
 * id: identifier of the projector object as stored in the astra-library.
 * projection_index: index of the projection angle
 * detector_index: index of the detector
 * weights: list of computed weights [pixel_index, weight]
 */
void astra_mex_projector_weights_single_ray(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: get input
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iPid = (int)(mxGetScalar(prhs[1]));
	int iProjectionIndex = (int)(mxGetScalar(prhs[2]));
	int iDetectorIndex = (int)(mxGetScalar(prhs[3]));

	// step2: get projector
	CProjector2D* pProjector = CProjector2DManager::getSingleton().get(iPid);
	if (!pProjector || !pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.\n");
		return;
	}
	
	// step3: create output vars
	int iStoredPixelCount;
	int iMaxPixelCount = pProjector->getProjectionWeightsCount(iProjectionIndex);
	SPixelWeight* pPixelsWeights = new SPixelWeight[iMaxPixelCount];
	
	// step4: perform operation
	pProjector->computeSingleRayWeights(iProjectionIndex, 
										iDetectorIndex, 
										pPixelsWeights, 
										iMaxPixelCount, 
										iStoredPixelCount);

	// step5: return output
	if (1 <= nlhs) {
		mwSize dims[2];
		dims[0] = iStoredPixelCount;
		dims[1] = 2;

		plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
		double* out = mxGetPr(plhs[0]);

		for (int col = 0; col < iStoredPixelCount; col++) {
			out[col] = pPixelsWeights[col].m_iIndex;
			out[iStoredPixelCount+col] = pPixelsWeights[col].m_fWeight;
		}	
	}
	
	// garbage collection
	delete[] pPixelsWeights;
}

//-----------------------------------------------------------------------------------------
/** weights = astra_mex_projector('weights_projection', id, projection_index);
 *
 * Calculate the nonzero weights of all rays in a certain projection.
 * id: identifier of the projector object as stored in the astra-library.
 * projection_index: index of the projection angle
 * weights: sparse matrix containing the rows of the projection matric belonging to the requested projection angle.
 */
void astra_mex_projector_weights_projection(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: get input
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iPid = (int)(mxGetScalar(prhs[1]));
	int iProjectionIndex = (int)(mxGetScalar(prhs[2]));

	// step2: get projector
	CProjector2D* pProjector = CProjector2DManager::getSingleton().get(iPid);
	if (!pProjector || !pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.\n");
		return;
	}

	// step3: create output vars
	SPixelWeight* pPixelsWheights = new SPixelWeight[pProjector->getProjectionWeightsCount(iProjectionIndex)];
	int* piRayStoredPixelCount = new int[pProjector->getProjectionGeometry()->getDetectorCount()];

	// step4: perform operation
	pProjector->computeProjectionRayWeights(iProjectionIndex, pPixelsWheights, piRayStoredPixelCount);

	// step5: return output
	if (1 <= nlhs) {
		// get basic values
		int iMatrixSize = pProjector->getVolumeGeometry()->getWindowLengthX() *
						  pProjector->getVolumeGeometry()->getWindowLengthY();
		int iDetectorCount = pProjector->getProjectionGeometry()->getDetectorCount();
		int iTotalStoredPixelCount = 0;
		for (int i = 0; i < iDetectorCount; i++) {
			iTotalStoredPixelCount += piRayStoredPixelCount[i];
		}

		// create matlab sparse matrix
		plhs[0] = mxCreateSparse(iMatrixSize,				// number of rows (#pixels)
							     iDetectorCount,			// number of columns (#detectors)
								 iTotalStoredPixelCount,	// number of non-zero elements
								 mxREAL);					// element type
		double* values = mxGetPr(plhs[0]);
		mwIndex* rows = mxGetIr(plhs[0]);
		mwIndex* cols = mxGetJc(plhs[0]);
		
		int currentBase = 0;
		int currentIndex = 0;
		for (int i = 0; i < iDetectorCount; i++) {
			for (int j = 0; j < piRayStoredPixelCount[i]; j++) {
				values[currentIndex + j] = pPixelsWheights[currentBase + j].m_fWeight;
				rows[currentIndex + j] = pPixelsWheights[currentBase + j].m_iIndex;
			}
					
			currentBase += pProjector->getProjectionWeightsCount(iProjectionIndex) / pProjector->getProjectionGeometry()->getDetectorCount();
			currentIndex += piRayStoredPixelCount[i];
		}
		cols[0] = piRayStoredPixelCount[0];
		for (int j = 1; j < iDetectorCount; j++) {
			cols[j] = cols[j-1] + piRayStoredPixelCount[j];
		}
		cols[iDetectorCount] = iTotalStoredPixelCount;
	}
	
	delete[] pPixelsWheights;
	delete[] piRayStoredPixelCount;
}

//-----------------------------------------------------------------------------------------
/** matrix_id = astra_mex_projector('matrix', id);
 *
 * Create an explicit projection matrix for this projector.
 * It returns an ID usable with astra_mex_matrix().
 * id: identifier of the projector object as stored in the astra-library.
 */
void astra_mex_projector_matrix(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: get input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iPid = (int)(mxGetScalar(prhs[1]));

	// step2: get projector
	CProjector2D* pProjector = CProjector2DManager::getSingleton().get(iPid);
	if (!pProjector || !pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.\n");
		return;
	}

	CSparseMatrix* pMatrix = pProjector->getMatrix();
	if (!pMatrix || !pMatrix->isInitialized()) {
		mexErrMsgTxt("Couldn't initialize data object.\n");
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
/** result = astra_mex_projector('is_cuda', id);
 *
 * Return is the specified projector is a cuda projector.
 * id: identifier of the projector object as stored in the astra-library.
 */
void astra_mex_projector_is_cuda(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	// step1: get input
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	int iPid = (int)(mxGetScalar(prhs[1]));

	// step2: get projector
	CProjector2D* pProjector = CProjector2DManager::getSingleton().get(iPid);
	if (!pProjector || !pProjector->isInitialized()) {
		mexErrMsgTxt("Projector not initialized.\n");
		return;
	}

#ifdef ASTRA_CUDA
	CCudaProjector2D* pCP = dynamic_cast<CCudaProjector2D*>(pProjector);
	plhs[0] = mxCreateLogicalScalar(pCP ? 1 : 0);
#else
	plhs[0] = mxCreateLogicalScalar(0);
#endif
}



//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: create, delete, clear, info, projection_geometry,\n");
	mexPrintf("             volume_geometry, weights_single_ray, weights_projection\n");
	mexPrintf("             matrix, is_cuda\n");
}


//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_projector(mode, ...);
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
		astra_mex_projector_create(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "delete") {
		astra_mex_projector_delete(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "clear") {
		astra_mex_projector_clear(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "info") {
		astra_mex_projector_info(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "projection_geometry") {
		astra_mex_projector_projection_geometry(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "volume_geometry") {
		astra_mex_projector_volume_geometry(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "weights_single_ray") {
		astra_mex_projector_weights_single_ray(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "weights_projection") {
		astra_mex_projector_weights_projection(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "matrix") {
		astra_mex_projector_matrix(nlhs, plhs, nrhs, prhs);
	} else if (sMode == "is_cuda") {
		astra_mex_projector_is_cuda(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}
	return;
}


