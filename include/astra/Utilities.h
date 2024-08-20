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

#ifndef _INC_ASTRA_UTILIES
#define _INC_ASTRA_UTILIES

#include <string>
#include <vector>
#include <map>
#include <cstdarg>

#include "Globals.h"

namespace astra {



namespace StringUtil {

// Exception thrown by functions below
class bad_cast : public std::exception {
public:
	bad_cast() { }
};

//< Format a string, returning it as a std::string
std::string vformat(const char *fmt, va_list ap);
//< Format a string, returning it as a std::string
std::string format(const char *fmt, ...) ATTRIBUTE_FORMAT(printf, 1, 2);


//< Parse string as int.
//< Throw exception on failure.
_AstraExport int stringToInt(const std::string& s);

//< Parse string as int.
//< Return fallback on failure.
_AstraExport int stringToInt(const std::string& s, int fallback);

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


_AstraExport void splitString(std::vector<std::string> &items, const std::string& s, const char *delim);

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
