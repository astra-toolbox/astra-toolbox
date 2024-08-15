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

#include "astra/XMLConfig.h"

#include "astra/Logging.h"

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


//-----------------------------------------------------------------------------

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
