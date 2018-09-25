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

#ifndef _INC_ASTRA_MEX_HELPFUNCTIONS
#define _INC_ASTRA_MEX_HELPFUNCTIONS


#define USE_MATLAB_UNDOCUMENTED


#include <string>
#include <list>
#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>
#include <mex.h>

#include <boost/any.hpp>

#include "astra/Globals.h"
#include "astra/Utilities.h"

#include "astra/Config.h"
#include "astra/XMLDocument.h"
#include "astra/XMLNode.h"

// utility functions
std::string mexToString(const mxArray* pInput);
bool mexIsScalar(const mxArray* pInput);
void get3DMatrixDims(const mxArray* x, mwSize *dims);

// convert boost::any into a MALTAB object
mxArray* vectorToMxArray(std::vector<astra::float32> mInput);
mxArray* anyToMxArray(boost::any _any);

// turn a MATLAB struct into a Config object
astra::Config* structToConfig(std::string rootname, const mxArray* pStruct);
bool structToXMLNode(astra::XMLNode node, const mxArray* pStruct);
bool optionsToXMLNode(astra::XMLNode node, const mxArray* pOptionStruct);
std::map<std::string, mxArray*> parseStruct(const mxArray* pInput);

// turn a Config object into a MATLAB struct
mxArray* configToStruct(astra::Config* cfg);
mxArray* XMLNodeToStruct(astra::XMLNode xml);
mxArray* stringToMxArray(std::string input);
mxArray* buildStruct(std::map<std::string, mxArray*> mInput);

#endif
