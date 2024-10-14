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



}
