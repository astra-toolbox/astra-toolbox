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

#include "astra/Algorithm.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CAlgorithm::CAlgorithm() : m_bShouldAbort(false), configCheckData(0) {
	
}

//----------------------------------------------------------------------------------------
// Destructor
CAlgorithm::~CAlgorithm() {

}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CAlgorithm::getInformation() 
{
	map<string, boost::any> result;
	result["Initialized"] = getInformation("Initialized");
	return result;
};

//----------------------------------------------------------------------------------------
// Information - Specific
boost::any CAlgorithm::getInformation(std::string _sIdentifier)
{
	if (_sIdentifier == "Initialized") { return m_bIsInitialized ? "yes" : "no"; } 
	return std::string("not found");
}

} // namespace astra
