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

#include "astra/Utilities.h"

using namespace std;
using namespace astra;

//-----------------------------------------------------------------------------
// Trim Whitespace Characters
void StringUtil::trim(std::string& _sString, bool _bLeft, bool _bRight)
{
	// trim right
	if (_bRight)
		_sString.erase(_sString.find_last_not_of(" \t\r") + 1); 

	// trim left
	if (_bLeft)
		_sString.erase(0, _sString.find_first_not_of(" \t\r")); 
}
//-----------------------------------------------------------------------------
// Split String
vector<string> StringUtil::split(const string& _sString, const string& _sDelims)
{
	std::vector<string> ret;

	size_t start, pos;
	start = 0;
	do {
		pos = _sString.find_first_of(_sDelims, start);
		if (pos == start) {
			// Do nothing
			start = pos + 1;
		} else if (pos == string::npos) {
			// Copy the rest of the string
			ret.push_back(_sString.substr(start));
			break;
		} else {
			// Copy up to newt delimiter
			ret.push_back(_sString.substr(start, pos - start));
			start = pos + 1;
		}

		// Parse up to next real data (in case there are two delims after each other)
		start = _sString.find_first_not_of(_sDelims, start);
	} while (pos != string::npos);

	return ret;
}
//-----------------------------------------------------------------------------
// Cast string to int
bool StringUtil::toInt(const string& _sString, int& _iValue)
{
	std::istringstream ss(_sString);
	ss >> _iValue;
	return !ss.fail();
}
//-----------------------------------------------------------------------------
// Cast string to float
bool StringUtil::toFloat32(const string& _sString, float32& _fValue)
{
	std::istringstream ss(_sString);
	ss >> _fValue;
	return !ss.fail();
}
//-----------------------------------------------------------------------------
// Convert string to Lower Case
void StringUtil::toLowerCase(std::string& _sString)
{
	std::transform(_sString.begin(),
				   _sString.end(),		
				   _sString.begin(),
				   ::tolower);
}
//-----------------------------------------------------------------------------    
// Convert string to Upper Case
void StringUtil::toUpperCase(std::string& _sString) 
{
	std::transform(_sString.begin(),
				   _sString.end(),
				   _sString.begin(),
				   ::toupper);
}
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------    
// Get Extension
string FileSystemUtil::getExtension(string& _sFilename)
{
	string sExtension = "";
	for (int i = _sFilename.length() - 1; 0 < i; i--) {
		if (_sFilename[i] == '.') {
			std::transform(sExtension.begin(),sExtension.end(),sExtension.begin(),::tolower);
			return sExtension;
		}
		sExtension = _sFilename[i] + sExtension;
	}
	return "";
}
//-----------------------------------------------------------------------------
