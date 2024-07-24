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

using namespace std;

namespace astra {

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
void Config::initialize(const std::string &rootname)
{
	if (!self) {
		XMLDocument* doc = XMLDocument::createDocument(rootname);
		self = doc->getRootNode();		
		_doc = doc;
	}
}


//-----------------------------------------------------------------------------
template <class T>
ConfigReader<T>::ConfigReader(const char *_name, T* _obj, const Config& _cfg)
	: object(_obj), cfg(&_cfg), objName(_name)
{
	assert(object);
	assert(cfg);
	assert(cfg->self);
	if (!object->configCheckData) {
		object->configCheckData = new ConfigCheckData;
		object->configCheckData->parseDepth = 0;
	}

	object->configCheckData->parseDepth++;
}

template <class T>
ConfigReader<T>::~ConfigReader()
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
bool ConfigReader<T>::stopParsing()
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
		os << objName << ": unused configuration options: " << errors;
		ASTRA_WARN(os.str().c_str());
		return false;
	}

	return true;
}

template <class T>
void ConfigReader<T>::markNodeParsed(const std::string& nodeName)
{
	assert(object->configCheckData);
	assert(object->configCheckData->parseDepth > 0);
	object->configCheckData->parsedNodes.insert(nodeName);
}

template <class T>
void ConfigReader<T>::markOptionParsed(const std::string& nodeName)
{
	assert(object->configCheckData);
	assert(object->configCheckData->parseDepth > 0);
	object->configCheckData->parsedOptions.insert(nodeName);
}


template<class T>
bool ConfigReader<T>::has(const std::string &name)
{
	XMLNode node = cfg->self.getSingleNode(name);
	return node;
}

template<class T>
bool ConfigReader<T>::getRequiredInt(const std::string &name, int &iValue)
{
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}

	try {
		iValue = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be an integer.", objName, name.c_str());
		return false;
	}

	markNodeParsed(name);

	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredNumerical(const std::string &name, float &fValue)
{
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}

	try {
		fValue = node.getContentNumerical();
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be numerical.", objName, name.c_str());
		return false;
	}

	markNodeParsed(name);

	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredID(const std::string &name, int &iValue)
{
	iValue = -1;
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		iValue = -1;
		return false;
	}

	bool ret = true;
	try {
		iValue = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: Unable to parse %s as an ID.", objName, name.c_str());
		iValue = -1;
		ret = false;
	}

	markNodeParsed(name);
	return ret;

}

template<class T>
bool ConfigReader<T>::getRequiredNumericalArray(const std::string &name, std::vector<double> &values)
{
	values.clear();
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}

	try {
		values = node.getContentNumericalArrayDouble();
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be a numerical matrix.", objName, name.c_str());
		return false;
	}

	markNodeParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredIntArray(const std::string &name, std::vector<int> &values)
{
	values.clear();
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}

	std::vector<std::string> data = node.getContentArray();
	values.resize(data.size());
	try {
		for (size_t i = 0; i < data.size(); ++i)
			values[i] = StringUtil::stringToInt(data[i]);
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be an integer array.", objName, name.c_str());
		return false;
	}

	markNodeParsed(name);
	return true;
}


template<class T>
bool ConfigReader<T>::getRequiredString(const std::string &name, std::string &sValue)
{
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	} else {
		sValue = node.getContent();
		markNodeParsed(name);
		return true;
	}

}

template<class T>
bool ConfigReader<T>::getID(const std::string &name, int &iValue)
{
	iValue = -1;
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		iValue = -1;
		return false;
	}

	bool ret = true;
	try {
		iValue = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		iValue = -1;
		ret = false;
	}

	markNodeParsed(name);
	return ret;

}

template<class T>
bool ConfigReader<T>::getString(const std::string &name, std::string &sValue, const std::string &sDefaultValue)
{
	XMLNode node = cfg->self.getSingleNode(name);
	if (!node) {
		sValue = sDefaultValue;
		return false;
	} else {
		sValue = node.getContent();
		markNodeParsed(name);
		return true;
	}
}



template<class T>
bool ConfigReader<T>::getRequiredSubConfig(const std::string &name, Config &_cfg, std::string &type)
{
	XMLNode node;
	node = cfg->self.getSingleNode(name);
	if (!node) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}

	type = node.getAttribute("type", "");

	_cfg = Config(node);

	markNodeParsed(name);

	return true;
}

template<class T>
bool ConfigReader<T>::hasOption(const std::string &name)
{
	return cfg->self.hasOption(name);
}


template<class T>
bool ConfigReader<T>::getOptionNumerical(const std::string &name, float &fValue, float fDefaultValue)
{
	fValue = fDefaultValue;
	try {
		fValue = cfg->self.getOptionNumerical(name, fDefaultValue);
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be numerical.", objName, name.c_str());
		return false;
	}

	markOptionParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getOptionInt(const std::string &name, int &iValue, int iDefaultValue)
{
	iValue = iDefaultValue;
	try {
		iValue = cfg->self.getOptionInt(name, iDefaultValue);
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be integer.", objName, name.c_str());
		return false;
	}

	markOptionParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getOptionUInt(const std::string &name, unsigned int &iValue, unsigned int iDefaultValue)
{
	iValue = iDefaultValue;
	try {
		int tmp = cfg->self.getOptionInt(name, iDefaultValue);
		if (tmp < 0) // HACK
			throw StringUtil::bad_cast();
		iValue = tmp;
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be unsigned integer.", objName, name.c_str());
		return false;
	}

	markOptionParsed(name);
	return true;
}


template<class T>
bool ConfigReader<T>::getOptionBool(const std::string &name, bool &bValue, bool bDefaultValue)
{
	bValue = bDefaultValue;
	try {
		bValue = cfg->self.getOptionBool(name, bDefaultValue);
	} catch (const StringUtil::bad_cast &e) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be boolean.", objName, name.c_str());
		return false;
	}

	markOptionParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getOptionString(const std::string &name, std::string &sValue, std::string sDefaultValue)
{
	sValue = cfg->self.getOption(name, sDefaultValue);

	markOptionParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getOptionIntArray(const std::string &name, std::vector<int> &values)
{
	values.clear();

	std::list<XMLNode> nodes = cfg->self.getNodes("Option");
	for (XMLNode &it : nodes) {
		if (it.getAttribute("key") == name) {
			std::vector<std::string> data = it.getContentArray();
			values.resize(data.size());
			try {
				for (size_t i = 0; i < data.size(); ++i)
					values[i] = StringUtil::stringToInt(data[i]);
			} catch (const StringUtil::bad_cast &e) {
				astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be an integer array.", objName, name.c_str());
				return false;
			}
			markNodeParsed(name);
			return true;
		}
	}

	return true;
}



template<class T>
bool ConfigReader<T>::getOptionID(const std::string &name, int &iValue)
{
	iValue = -1;

	if (!cfg->self.hasOption(name))
		return false;

	bool ret = true;
	try {
		iValue = cfg->self.getOptionInt(name, -1);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_WARN("Optional parameter %s is not a valid id", name.c_str());
		iValue = -1;
		ret = false;
	}

	markOptionParsed(name);
	return ret;
}


template class ConfigReader<CAlgorithm>;
template class ConfigReader<CProjectionGeometry2D>;
template class ConfigReader<CProjectionGeometry3D>;
template class ConfigReader<CVolumeGeometry2D>;
template class ConfigReader<CVolumeGeometry3D>;
template class ConfigReader<CProjector2D>;
template class ConfigReader<CProjector3D>;


}
