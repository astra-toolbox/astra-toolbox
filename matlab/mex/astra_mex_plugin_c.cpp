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

/** \file astra_mex_plugin_c.cpp
 *
 *  \brief Manages Python plugins.
 */

#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include "astra/PluginAlgorithmFactory.h"

#include <Python.h>

using namespace std;
using namespace astra;

static void fixLapackLoading()
{
    // When running in Matlab, we need to force numpy
    // to use its internal lapack library instead of
    // Matlab's MKL library to avoid errors. To do this,
    // we set Python's dlopen flags to RTLD_NOW|RTLD_DEEPBIND
    // and import 'numpy.linalg.lapack_lite' here. We reset
    // Python's dlopen flags afterwards.
    PyObject *sys = PyImport_ImportModule("sys");
    if (sys != NULL) {
        PyObject *curFlags = PyObject_CallMethod(sys, "getdlopenflags", NULL);
        if (curFlags != NULL) {
            PyObject *retVal = PyObject_CallMethod(sys, "setdlopenflags", "i", 10); // RTLD_NOW|RTLD_DEEPBIND
            if (retVal != NULL) {
                PyObject *lapack = PyImport_ImportModule("numpy.linalg.lapack_lite");
                if (lapack != NULL) {
                    Py_DECREF(lapack);
                }
                PyObject *retVal2 = PyObject_CallMethod(sys, "setdlopenflags", "O",curFlags);
                if (retVal2 != NULL) {
                    Py_DECREF(retVal2);
                }
                Py_DECREF(retVal);
            }
            Py_DECREF(curFlags);
        }
        Py_DECREF(sys);
    }
}

//-----------------------------------------------------------------------------------------
/** astra_mex_plugin('init');
 *
 * Initialize plugin support by initializing python and importing astra
 */
void astra_mex_plugin_init()
{
    if(!Py_IsInitialized()){
        Py_Initialize();
        PyEval_InitThreads();
    }

#ifndef _MSC_VER
    fixLapackLoading();
#endif

    // Importing astra may be overkill, since we only need to initialize
    // PythonPluginAlgorithmFactory from astra.plugin_c.
    PyObject *mod = PyImport_ImportModule("astra");
    Py_XDECREF(mod);
}


//-----------------------------------------------------------------------------------------
/** astra_mex_plugin('get_registered');
 *
 * Print registered plugins.
 */
void astra_mex_plugin_get_registered(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    astra::CPluginAlgorithmFactory *fact = astra::CPluginAlgorithmFactory::getFactory();
    if (!fact) {
        mexPrintf("Plugin support not initialized.");
        return;
    }
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
    astra::CPluginAlgorithmFactory *fact = astra::CPluginAlgorithmFactory::getFactory();
    if (!fact) {
        mexPrintf("Plugin support not initialized.");
        return;
    }
    if (2 <= nrhs) {
        string class_name = mexToString(prhs[1]);
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
    astra::CPluginAlgorithmFactory *fact = astra::CPluginAlgorithmFactory::getFactory();
    if (!fact) {
        mexPrintf("Plugin support not initialized.");
        return;
    }
    if (2 <= nrhs) {
        string name = mexToString(prhs[1]);
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
	if (sMode == "init") {
		astra_mex_plugin_init();
	} else if (sMode ==  std::string("get_registered")) {
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


