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

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <sstream>
#include <locale>
#include <iomanip>

namespace astra {

namespace StringUtil {

int stringToInt(const std::string& s)
{
	int i;
	std::istringstream iss(s);
	iss.imbue(std::locale::classic());
	iss >> i;
	if (iss.fail() || !iss.eof())
		throw bad_cast();
	return i;

}

float stringToFloat(const std::string& s)
{
	return (float)stringToDouble(s);
}

double stringToDouble(const std::string& s)
{
	double f;
	std::istringstream iss(s);
	iss.imbue(std::locale::classic());
	iss >> f;
	if (iss.fail() || !iss.eof())
		throw bad_cast();
	return f;
}

template<> float stringTo(const std::string& s) { return stringToFloat(s); }
template<> double stringTo(const std::string& s) { return stringToDouble(s); }

std::vector<float> stringToFloatVector(const std::string &s)
{
	return stringToVector<float>(s);
}

std::vector<double> stringToDoubleVector(const std::string &s)
{
	return stringToVector<double>(s);
}

template<typename T>
std::vector<T> stringToVector(const std::string& s)
{
	// split
	std::vector<std::string> items;
	boost::split(items, s, boost::is_any_of(",;"));

	// init list
	std::vector<T> out;
	out.resize(items.size());

	// loop elements
	for (unsigned int i = 0; i < items.size(); i++) {
		out[i] = stringTo<T>(items[i]);
	}
	return out;
}


std::string floatToString(float f)
{
	std::ostringstream s;
	s.imbue(std::locale::classic());
	s << std::setprecision(9) << f;
	return s.str();
}

std::string doubleToString(double f)
{
	std::ostringstream s;
	s.imbue(std::locale::classic());
	s << std::setprecision(17) << f;
	return s.str();
}


template<> std::string toString(float f) { return floatToString(f); }
template<> std::string toString(double f) { return doubleToString(f); }


}

}
