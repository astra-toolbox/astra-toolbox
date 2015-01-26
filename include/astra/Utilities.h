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
#include <algorithm>
#include <sstream>
#include <map>

#include "Globals.h"

namespace astra {

/**
 * This class contains some usefull static utility functions for std strings.
 */
class StringUtil {

public:
	/**
	 * Removes whitespace characters such as spaces and tabs at the extremas.
	 * Optionally you can specify which extrema to trim (default=both) 
	 *
	 * @param _sString The string to trim.
	 * @param _bLeft Trim the left extrema?  Default = true.
	 * @param _bRight Trim the right extrema?  Default = true.
	 */
	static void trim(std::string& _sString, bool _bLeft = true, bool _bRight = true);

	/**
	 * Returns a vector of strings that contains all the substrings delimited by 
	 * the characters in _sDelims.
	 *
	 * @param _sString The string to split.
	 * @param _sDelims The delimiter string.
	 * @return Vector of strings.
	 */
	static std::vector<std::string> split(const std::string& _sString, const std::string& _sDelims);

	/**
	 * Cast a string to an integer.
	 *
	 * @param _sString The string to cast.
	 * @param _iValue Output integer parameter.
	 * @return success?
	 */
	static bool toInt(const std::string& _sString, int& _iValue);

	/**
	 * Cast a string to a float32.
	 *
	 * @param _sString The string to cast.
	 * @param _fValue Output float32 parameter.
	 * @return success?
	 */
	static bool toFloat32(const std::string& _sString, float32& _fValue);

	/**
	 * Convert a string to lower case.
	 *
	 * @param _sString The string to convert.
	 */
	static void toLowerCase(std::string& _sString);
	
	/**
	 * Convert a string to upper case.
	 *
	 * @param _sString The string to convert.
	 */
	static void toUpperCase(std::string& _sString);
};

/**
 * This class contains some usefull static utility functions for std strings.
 */
class FileSystemUtil {

public:
	/**
	 * Get the extensions of a filename.  Always in lower case.
	 *
	 * @param _sFilename file to get extensions from.
	 * @return Extension (lower case).  Empty string if filename is a directory or not a valid file format.
	 */
	static std::string getExtension(std::string& _sFilename);


};


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
