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

#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include "astra/Globals.h"

using namespace std;
using namespace astra;


//-----------------------------------------------------------------------------------------
/** astra_mex('credits');
 * 
 * Print Credits
 */
void astra_mex_credits(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	cout << "All Scale Tomographic Reconstruction Antwerp Toolbox (ASTRA-Toolbox) was developed at the University of Antwerp by" << endl;
	cout << " * Joost Batenburg, PhD" << endl;
	cout << " * Gert Merckx" << endl;
	cout << " * Willem Jan Palenstijn" << endl;
	cout << " * Tom Roelandts" << endl;
	cout << " * Prof. Dr. Jan Sijbers" << endl;
	cout << " * Wim van Aarle" << endl;
	cout << " * Sander van der Maar" << endl;
	cout << " * Gert Van Gompel, PhD" << endl;
}

//-----------------------------------------------------------------------------------------
/** use_cuda = astra_mex('use_cuda');
 * 
 * Is CUDA enabled?
 */
void astra_mex_use_cuda(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{ 
	if (1 <= nlhs) {
		plhs[0] = mxCreateDoubleScalar(astra::cudaEnabled() ? 1 : 0);
	}
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
		cout << "astra toolbox version " << astra::getVersionString() << endl;
	}
}

//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("   Valid modes: version, use_cuda, credits\n");
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
		sMode = mex_util_get_string(prhs[0]);	
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
	} else {
		printHelp();
	}

	return;
}


