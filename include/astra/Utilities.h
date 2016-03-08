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

#ifndef _INC_ASTRA_UTILIES
#define _INC_ASTRA_UTILIES

#include <string>
#include <vector>
#include <map>

#include "Globals.h"

namespace astra {



namespace StringUtil {

// Exception thrown by functions below
class bad_cast : public std::exception {
public:
	bad_cast() { }
};


//< Parse string as int.
//< Throw exception on failure.
_AstraExport int stringToInt(const std::string& s);

//< Parse string as float.
//< Throw exception on failure.
_AstraExport float stringToFloat(const std::string& s);

//< Parse string as double.
//< Throw exception on failure.
_AstraExport double stringToDouble(const std::string& s);

template<typename T>
_AstraExport T stringTo(const std::string& s);

//< Parse comma/semicolon-separated string as float vector.
//< Throw exception on failure.
_AstraExport std::vector<float> stringToFloatVector(const std::string& s);

//< Parse comma/semicolon-separated string as double vector.
//< Throw exception on failure.
_AstraExport std::vector<double> stringToDoubleVector(const std::string& s);

template<typename T>
_AstraExport std::vector<T> stringToVector(const std::string& s);



//< Generate string from float.
_AstraExport std::string floatToString(float f);

//< Generate string from double.
_AstraExport std::string doubleToString(double f);

template<typename T>
_AstraExport std::string toString(T f);

}




template<typename T, typename S>
std::map<T,S> mergeMap(std::map<T,S> _mMap1, std::map<T,S> _mMap2) 
{
	std::map<T,S> result = _mMap1;
	for (typename std::map<T,S>::iterator it = _mMap2.begin(); it != _mMap2.end(); it++) {
		result[(*it).first] = (*it).second;
	}
	return result;
}

} // end namespace

#endif
