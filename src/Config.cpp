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

XMLConfig::XMLConfig(XMLNode _self)
{
	self = _self;
	_doc = 0;
}

XMLConfig::XMLConfig(const std::string &rootname)
{
	_doc = XMLDocument::createDocument(rootname);
	self = _doc->getRootNode();
}

XMLConfig::~XMLConfig()
{
	delete _doc;
	_doc = 0;
}


//-----------------------------------------------------------------------------

//virtual
bool XMLConfig::has(const std::string &name) const
{
	XMLNode node = self.getSingleNode(name);
	if (!node)
		return false;
	return true;
}

bool XMLConfig::hasOption(const std::string &name) const
{
	return self.hasOption(name);
}


bool XMLConfig::getSubConfig(const std::string &name, Config *&_cfg, std::string &type) const
{
	XMLNode node;
	node = self.getSingleNode(name);
	if (!node)
		return false;

	type = node.getAttribute("type", "");
	_cfg = new XMLConfig(node);

	return true;
}


bool XMLConfig::getInt(const std::string &name, int &iValue) const
{
	XMLNode node = self.getSingleNode(name);
	if (!node)
		return false;
	try {
		iValue = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	return true;
}

bool XMLConfig::getFloat(const std::string &name, float &fValue) const
{
	XMLNode node = self.getSingleNode(name);
	if (!node)
		return false;
	try {
		fValue = node.getContentNumerical();
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	return true;

}

bool XMLConfig::getDoubleArray(const std::string &name, std::vector<double> &values) const
{
	values.clear();
	XMLNode node = self.getSingleNode(name);
	if (!node)
		return false;
	try {
		values = node.getContentNumericalArrayDouble();
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	return true;

}

bool XMLConfig::getIntArray(const std::string &name, std::vector<int> &values) const
{
	values.clear();
	XMLNode node = self.getSingleNode(name);
	if (!node)
		return false;
	// TODO: Don't go via doubles
	std::vector<double> tmp;
	try {
		tmp = node.getContentNumericalArrayDouble();
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	values.resize(tmp.size());
	for (size_t i = 0; i < tmp.size(); ++i) {
		int t = static_cast<int>(tmp[i]);
		if (t != tmp[i])
			return false;
		values[i] = t;
	}
	return true;
}

bool XMLConfig::getString(const std::string &name, std::string &sValue) const
{
	XMLNode node = self.getSingleNode(name);
	if (!node)
		return false;
	sValue = node.getContent();
	return true;
}


bool XMLConfig::getOptionFloat(const std::string &name, float &fValue) const
{
	try {
		fValue = self.getOptionNumerical(name);
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	return true;
}

bool XMLConfig::getOptionInt(const std::string &name, int &iValue) const
{
	try {
		iValue = self.getOptionInt(name);
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	return true;
}

bool XMLConfig::getOptionUInt(const std::string &name, unsigned int &iValue) const
{
	int tmp = 0;
	try {
		tmp = self.getOptionInt(name);
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	if (tmp < 0)
		return false;
	iValue = (unsigned int)tmp;
	return true;
}

bool XMLConfig::getOptionBool(const std::string &name, bool &bValue) const
{
	try {
		bValue = self.getOptionBool(name);
	} catch (const StringUtil::bad_cast &e) {
		return false;
	}
	return true;
}

bool XMLConfig::getOptionString(const std::string &name, std::string &sValue) const
{
	sValue = self.getOption(name);
	return true;
}

bool XMLConfig::getOptionIntArray(const std::string &name, std::vector<int> &values) const
{
	values.clear();

	std::list<XMLNode> nodes = self.getNodes("Option");
	for (XMLNode &it : nodes) {
		if (it.getAttribute("key") == name) {
			std::vector<std::string> data = it.getContentArray();
			values.resize(data.size());
			try {
				for (size_t i = 0; i < data.size(); ++i)
					values[i] = StringUtil::stringToInt(data[i]);
			} catch (const StringUtil::bad_cast &e) {
				return false;
			}
			return true;
		}
	}
	return false;
}




//-----------------------------------------------------------------------------
template <class T>
ConfigReader<T>::ConfigReader(const char *_name, T* _obj, const Config& _cfg)
	: object(_obj), cfg(&_cfg), objName(_name)
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

std::list<std::string> XMLConfig::checkUnparsed(const ConfigCheckData &data) const
{
	std::list<std::string> errors;

	for (XMLNode &i : self.getNodes()) {
		std::string nodeName = i.getName();

		if (nodeName == "Option") {
			nodeName = i.getAttribute("key", "");
			if (data.parsedOptions.find(nodeName) == data.parsedOptions.end()) {
				errors.push_back(nodeName);
			}
		} else {
			if (data.parsedNodes.find(nodeName) == data.parsedNodes.end()) {
				errors.push_back(nodeName);
			}
		}
	}

	return errors;
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

	std::list<std::string> errors = cfg->checkUnparsed(*object->configCheckData);

	if (!errors.empty()) {
		ostringstream os;
		os << objName << ": unused configuration options: ";

		os << errors.front();
		errors.pop_front();
		for (const std::string &str : errors)
			os << str;

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
	return cfg->has(name);
}

template<class T>
bool ConfigReader<T>::getRequiredInt(const std::string &name, int &iValue)
{
	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getInt(name, iValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be an integer.", objName, name.c_str());
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredNumerical(const std::string &name, float &fValue)
{
	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getFloat(name, fValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be numerical.", objName, name.c_str());
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredID(const std::string &name, int &iValue)
{
	iValue = -1;

	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getInt(name, iValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: Unable to parse %s as an ID.", objName, name.c_str());
		iValue = -1;
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredNumericalArray(const std::string &name, std::vector<double> &values)
{
	values.clear();
	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getDoubleArray(name, values)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be a numerical matrix.", objName, name.c_str());
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::getRequiredIntArray(const std::string &name, std::vector<int> &values)
{
	values.clear();
	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getIntArray(name, values)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be an integer array.", objName, name.c_str());
		return false;
	}
	return true;
}


template<class T>
bool ConfigReader<T>::getRequiredString(const std::string &name, std::string &sValue)
{
	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getString(name, sValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be a string.", objName, name.c_str());
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::getID(const std::string &name, int &iValue)
{
	iValue = -1;

	if (!cfg->has(name))
		return false;
	markNodeParsed(name);

	if (!cfg->getInt(name, iValue)) {
		iValue = -1;
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::getString(const std::string &name, std::string &sValue, const std::string &sDefaultValue)
{
	sValue = sDefaultValue;
	if (!cfg->has(name))
		return false;
	markNodeParsed(name);

	if (!cfg->getString(name, sValue))
		return false;

	return true;
}



template<class T>
bool ConfigReader<T>::getRequiredSubConfig(const std::string &name, Config *&_cfg, std::string &type)
{
	if (!cfg->has(name)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: No %s tag specified.", objName, name.c_str());
		return false;
	}
	markNodeParsed(name);

	if (!cfg->getSubConfig(name, _cfg, type)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: %s must be a configuration object.", objName, name.c_str());
		return false;
	}
	return true;
}

template<class T>
bool ConfigReader<T>::hasOption(const std::string &name)
{
	return cfg->hasOption(name);
}


template<class T>
bool ConfigReader<T>::getOptionNumerical(const std::string &name, float &fValue, float fDefaultValue)
{
	fValue = fDefaultValue;
	if (cfg->hasOption(name) && !cfg->getOptionFloat(name, fValue)) {
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
	if (cfg->hasOption(name) && !cfg->getOptionInt(name, iValue)) {
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
	if (cfg->hasOption(name) && !cfg->getOptionUInt(name, iValue)) {
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
	if (cfg->hasOption(name) && !cfg->getOptionBool(name, bValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be boolean.", objName, name.c_str());
		return false;
	}

	markOptionParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getOptionString(const std::string &name, std::string &sValue, std::string sDefaultValue)
{
	sValue = sDefaultValue;
	if (cfg->hasOption(name) && !cfg->getOptionString(name, sValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be a string.", objName, name.c_str());
		return false;
	}

	markOptionParsed(name);
	return true;
}

template<class T>
bool ConfigReader<T>::getOptionIntArray(const std::string &name, std::vector<int> &values)
{
	values.clear();
	if (cfg->hasOption(name) && !cfg->getOptionIntArray(name, values)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be an integer array.", objName, name.c_str());
		return false;
	}

	markNodeParsed(name);
	return true;
}



template<class T>
bool ConfigReader<T>::getOptionID(const std::string &name, int &iValue)
{
	iValue = -1;
	if (!cfg->hasOption(name))
		return false;

	if (!cfg->getOptionInt(name, iValue)) {
		astra::CLogger::error(__FILE__, __LINE__, "Configuration error in %s: option %s must be an ID.", objName, name.c_str());
		iValue = -1;
		return false;
	}

	markOptionParsed(name);
	return true;
}

// TODO: Add base class "Configurable"
template class ConfigReader<CAlgorithm>;
template class ConfigReader<CProjectionGeometry2D>;
template class ConfigReader<CProjectionGeometry3D>;
template class ConfigReader<CVolumeGeometry2D>;
template class ConfigReader<CVolumeGeometry3D>;
template class ConfigReader<CProjector2D>;
template class ConfigReader<CProjector3D>;



ConfigWriter::ConfigWriter(const std::string &name)
{
	cfg = new XMLConfig(name);
}

ConfigWriter::ConfigWriter(const std::string &name, const std::string &type)
	: ConfigWriter(name)
{
	cfg->self.addAttribute("type", type);
}


ConfigWriter::~ConfigWriter()
{
	delete cfg;
}

Config* ConfigWriter::getConfig()
{
	Config *ret = cfg;
	cfg = nullptr;

	return ret;
}

void ConfigWriter::addInt(const std::string &name, int iValue)
{
	cfg->self.addChildNode(name, iValue);
}

void ConfigWriter::addNumerical(const std::string &name, double fValue)
{
	cfg->self.addChildNode(name, fValue);
}

void ConfigWriter::addNumericalArray(const std::string &name, const float* pfValues, int iCount)
{
	XMLNode res = cfg->self.addChildNode(name);
	res.setContent(pfValues, iCount);
}

void ConfigWriter::addNumericalMatrix(const std::string &name, const double* pfValues, int iHeight, int iWidth)
{
	XMLNode res = cfg->self.addChildNode(name);
	res.setContent(pfValues, iWidth, iHeight, false);
}

void ConfigWriter::addID(const std::string &name, int iValue)
{
	cfg->self.addChildNode(name, iValue);
}

void ConfigWriter::addOptionNumerical(const std::string &name, double fValue)
{
	cfg->self.addOption(name, fValue);
}





}
