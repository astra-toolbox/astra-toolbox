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

#ifndef _INC_ASTRA_CONFIG
#define _INC_ASTRA_CONFIG

#include "Globals.h"
#include "XMLNode.h"
#include "XMLDocument.h"

#include <set>

namespace astra {


/**
 * Configuration options for an ASTRA class.
 */
struct _AstraExport Config {

	Config();
	Config(XMLNode _self);
	~Config();

	void initialize(std::string rootname);

	XMLNode self;

	XMLDocument *_doc;
};

struct ConfigCheckData {
	// For checking for unparsed nodes/options
	std::set<std::string> parsedNodes;
	std::set<std::string> parsedOptions;
	unsigned int parseDepth;
};


template<class T>
class ConfigStackCheck {
public:
	ConfigStackCheck(const char *_name, T* _obj, const Config& _cfg);
	~ConfigStackCheck();

	bool stopParsing(); // returns true if no unused nodes/options
	void markNodeParsed(const std::string& name);
	void markOptionParsed(const std::string& name);
	

private:
	T* object;
	const Config* cfg;
	const char* name;
};

} // end namespace

#endif
