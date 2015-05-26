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

#include "astra/Config.h"

// For explicit ConfigStackCheck instantiations
#include "astra/Algorithm.h"
#include "astra/VolumeGeometry2D.h"
#include "astra/VolumeGeometry3D.h"
#include "astra/ProjectionGeometry2D.h"
#include "astra/ProjectionGeometry3D.h"
#include "astra/Projector2D.h"
#include "astra/Projector3D.h"

#include "astra/Logging.h"
#include <sstream>

using namespace astra;
using namespace std;

//-----------------------------------------------------------------------------
// default constructor
Config::Config() : self()
{
	_doc = 0;
}

//-----------------------------------------------------------------------------
// not so default constructor
Config::Config(XMLNode _self)
{
	self = _self;
	_doc = 0;
}

//-----------------------------------------------------------------------------
Config::~Config()
{
	delete _doc;
	_doc = 0;
}

//-----------------------------------------------------------------------------
void Config::initialize(std::string rootname)
{
	if (!self) {
		XMLDocument* doc = XMLDocument::createDocument(rootname);
		self = doc->getRootNode();		
		_doc = doc;
	}
}


//-----------------------------------------------------------------------------
template <class T>
ConfigStackCheck<T>::ConfigStackCheck(const char *_name, T* _obj, const Config& _cfg)
	: object(_obj), cfg(&_cfg), name(_name)
{
	assert(object);
	assert(cfg);
	if (!object->configCheckData) {
		object->configCheckData = new ConfigCheckData;
		object->configCheckData->parseDepth = 0;
	}

	object->configCheckData->parseDepth++;
}

template <class T>
ConfigStackCheck<T>::~ConfigStackCheck()
{
	assert(object->configCheckData);
	assert(object->configCheckData->parseDepth > 0);


	if (object->configCheckData->parseDepth == 1) {
		// Entirely done with parsing this Config object

		if (object->isInitialized())
			stopParsing();

		delete object->configCheckData;
		object->configCheckData = 0;
	} else {
		object->configCheckData->parseDepth--;
	}
}


// returns true if no unused nodes/options
template <class T>
bool ConfigStackCheck<T>::stopParsing()
{
	assert(object->configCheckData);
	assert(object->configCheckData->parseDepth > 0);

	if (object->configCheckData->parseDepth > 1)
		return true;

	// If this was the top-level parse function, check

	std::string errors;

	std::list<XMLNode> nodes = cfg->self.getNodes();
	for (std::list<XMLNode>::iterator i = nodes.begin(); i != nodes.end(); ++i)
	{
		std::string nodeName = i->getName();

		if (nodeName == "Option") {
			nodeName = i->getAttribute("key", "");
			if (object->configCheckData->parsedOptions.find(nodeName) == object->configCheckData->parsedOptions.end()) {
				if (!errors.empty()) errors += ", ";
				errors += nodeName;
			}
		} else {
			if (object->configCheckData->parsedNodes.find(nodeName) == object->configCheckData->parsedNodes.end()) {
				if (!errors.empty()) errors += ", ";
				errors += nodeName;
			}
		}
	}
	nodes.clear();

	if (!errors.empty()) {
		ostringstream os;
		os << name << ": unused configuration options: " << errors;
		ASTRA_WARN(os.str().c_str());
		return false;
	}

	return true;
}

template <class T>
void ConfigStackCheck<T>::markNodeParsed(const std::string& nodeName)
{
	assert(object->configCheckData);
	assert(object->configCheckData->parseDepth > 0);
	object->configCheckData->parsedNodes.insert(nodeName);
}

template <class T>
void ConfigStackCheck<T>::markOptionParsed(const std::string& nodeName)
{
	assert(object->configCheckData);
	assert(object->configCheckData->parseDepth > 0);
	object->configCheckData->parsedOptions.insert(nodeName);
}


template class ConfigStackCheck<CAlgorithm>;
template class ConfigStackCheck<CProjectionGeometry2D>;
template class ConfigStackCheck<CProjectionGeometry3D>;
template class ConfigStackCheck<CVolumeGeometry2D>;
template class ConfigStackCheck<CVolumeGeometry3D>;
template class ConfigStackCheck<CProjector2D>;
template class ConfigStackCheck<CProjector3D>;

