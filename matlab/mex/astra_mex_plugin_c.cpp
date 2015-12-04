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

/** \file astra_mex_plugin_c.cpp
 *
 *  \brief Manages Python plugins.
 */

#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include "astra/PluginAlgorithm.h"

using namespace std;
using namespace astra;


//-----------------------------------------------------------------------------------------
/** astra_mex_plugin('get_registered');
 *
 * Print registered plugins.
 */
void astra_mex_plugin_get_registered(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    astra::CPluginAlgorithmFactory *fact = astra::CPluginAlgorithmFactory::getSingletonPtr();
    std::map<std::string, std::string> mp = fact->getRegisteredMap();
    for(std::map<std::string,std::string>::iterator it=mp.begin();it!=mp.end();it++){
        mexPrintf("%s: %s\n",it->first.c_str(), it->second.c_str());
    }
}

//-----------------------------------------------------------------------------------------
/** astra_mex_plugin('register', class_name);
 *
 * Register plugin.
 */
void astra_mex_plugin_register(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (2 <= nrhs) {
        string class_name = mexToString(prhs[1]);
        astra::CPluginAlgorithmFactory *fact = astra::CPluginAlgorithmFactory::getSingletonPtr();
        fact->registerPlugin(class_name);
    }else{
        mexPrintf("astra_mex_plugin('register', class_name);\n");
    }
}

//-----------------------------------------------------------------------------------------
/** astra_mex_plugin('get_help', name);
 *
 * Get help about plugin.
 */
void astra_mex_plugin_get_help(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (2 <= nrhs) {
        string name = mexToString(prhs[1]);
        astra::CPluginAlgorithmFactory *fact = astra::CPluginAlgorithmFactory::getSingletonPtr();
        mexPrintf((fact->getHelp(name)+"\n").c_str());
    }else{
        mexPrintf("astra_mex_plugin('get_help', name);\n");
    }
}


//-----------------------------------------------------------------------------------------

static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("   Valid modes: register, get_registered, get_help\n");
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
	if (sMode ==  std::string("get_registered")) { 
		astra_mex_plugin_get_registered(nlhs, plhs, nrhs, prhs); 
    }else if (sMode ==  std::string("get_help")) { 
        astra_mex_plugin_get_help(nlhs, plhs, nrhs, prhs); 
    }else if (sMode ==  std::string("register")) { 
		astra_mex_plugin_register(nlhs, plhs, nrhs, prhs); 
	} else {
		printHelp();
	}

	return;
}


