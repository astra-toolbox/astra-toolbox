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

#ifndef _INC_ASTRA_CONFIG
#define _INC_ASTRA_CONFIG

#include "Globals.h"
#include "XMLNode.h"
#include "XMLDocument.h"

#include <set>

namespace astra {

struct ConfigCheckData {
	// For checking for unparsed nodes/options
	std::set<std::string> parsedNodes;
	std::set<std::string> parsedOptions;
	unsigned int parseDepth;
};

/**
 * Configuration options for an ASTRA class.
 */
class _AstraExport Config {
public:
	Config() { }
	virtual ~Config() { }

	void initialize(const std::string &rootname);

	template<class TT> friend class ConfigReader;
	friend class ConfigWriter;

protected:
	virtual bool has(const std::string &name) const = 0;
	virtual bool hasOption(const std::string &name) const = 0;

	virtual bool getSubConfig(const std::string &name, Config *&_cfg, std::string &type) const = 0;

	virtual bool getInt(const std::string &name, int &iValue) const = 0;
	virtual bool getFloat(const std::string &name, float &fValue) const = 0;
	virtual bool getDoubleArray(const std::string &name, std::vector<double> &values) const = 0;
	virtual bool getIntArray(const std::string &name, std::vector<int> &values) const = 0;
	virtual bool getString(const std::string &name, std::string &sValue) const = 0;

	virtual bool getOptionFloat(const std::string &name, float &fValue) const = 0;
	virtual bool getOptionInt(const std::string &name, int &iValue) const = 0;
	virtual bool getOptionUInt(const std::string &name, unsigned int &iValue) const = 0;
	virtual bool getOptionBool(const std::string &name, bool &bValue) const = 0;
	virtual bool getOptionString(const std::string &name, std::string &sValue) const = 0;
	virtual bool getOptionIntArray(const std::string &name, std::vector<int> &values) const = 0;

	virtual std::list<std::string> checkUnparsed(const ConfigCheckData &data) const = 0;
};

class _AstraExport XMLConfig : public Config {
public:
	XMLConfig(XMLNode _node);
	XMLConfig(const std::string &rootname);

	virtual ~XMLConfig();
private:
	template<class TT> friend class ConfigReader;

	virtual bool has(const std::string &name) const;
	virtual bool hasOption(const std::string &name) const;

	virtual bool getSubConfig(const std::string &name, Config *&_cfg, std::string &type) const;

	virtual bool getInt(const std::string &name, int &iValue) const;
	virtual bool getFloat(const std::string &name, float &fValue) const;
	virtual bool getDoubleArray(const std::string &name, std::vector<double> &values) const;
	virtual bool getIntArray(const std::string &name, std::vector<int> &values) const;
	virtual bool getString(const std::string &name, std::string &sValue) const;

	virtual bool getOptionFloat(const std::string &name, float &fValue) const;
	virtual bool getOptionInt(const std::string &name, int &iValue) const;
	virtual bool getOptionUInt(const std::string &name, unsigned int &iValue) const;
	virtual bool getOptionBool(const std::string &name, bool &bValue) const;
	virtual bool getOptionString(const std::string &name, std::string &sValue) const;
	virtual bool getOptionIntArray(const std::string &name, std::vector<int> &values) const;

	virtual std::list<std::string> checkUnparsed(const ConfigCheckData &data) const;
private:
	friend class ConfigWriter;
	XMLDocument *_doc;
public:
	// TODO: Make this private once python/matlab interfaces can handle that
	XMLNode self;
};


template<class T>
class ConfigReader {
public:
	ConfigReader(const char *_name, T *_obj, const Config &_cfg);
	~ConfigReader();

	// Return true if config has a value
	bool has(const std::string &name);

	// Get and parse values, and return true if successful.
	// In case of missing values or parse errors, report the error and
	// return false.
	bool getRequiredInt(const std::string &name, int &iValue);
	bool getRequiredNumerical(const std::string &name, float &fValue);
	bool getRequiredID(const std::string &name, int &iValue);

	bool getRequiredNumericalArray(const std::string &name, std::vector<double> &values);
	bool getRequiredIntArray(const std::string &name, std::vector<int> &values);
	bool getRequiredString(const std::string &name, std::string &sValue);

	// Get a sub-configuration, and return true if succesful.
	// In case of missing values or parse errors, report the error and
	// return false.
	// For convenience, also directly get the "type" attribute of the subcfg.
	// If it has no type attribute, return empty string as type. (That is not
	// considered an error.)
	bool getRequiredSubConfig(const std::string &name, Config *&_cfg, std::string &type);

	// Get a value and parse it as an ID. Returns true if successful,
	// and false otherwise (returning -1 as iValue). Reports no errors.
	bool getID(const std::string &name, int &iValue);

	// Get a string value. Returns true if successful, and false otherwise
	// (return default value). Reports no errors.
	bool getString(const std::string &name, std::string &sValue, const std::string &sDefaultValue);

	// Return true if config has an option
	bool hasOption(const std::string &name);

	// Get and parse an option value. Returns true if the option is present
	// and successfully parsed. Returns true and the default value if the option
	// is not present. Returns false and the default value if the option
	// is present but malformed. Reports parsing errors.
	bool getOptionNumerical(const std::string &name, float &fValue, float fDefaultValue);
	bool getOptionInt(const std::string &name, int &iValue, int iDefaultValue);
	bool getOptionUInt(const std::string &name, unsigned int &iValue, unsigned int iDefaultValue);
	bool getOptionBool(const std::string &name, bool &bValue, bool bDefaultValue);
	bool getOptionString(const std::string &name, std::string &sValue, std::string sDefaultValue);
	bool getOptionIntArray(const std::string &name, std::vector<int> &values);

	// Get and parse an option value as ID. Returns true if the option is
	// present and successfully parsed. Returns false and -1 if the option is
	// not present or malformed. Reports parsing errors.
	bool getOptionID(const std::string &name, int &iValue);

private:
	T* object;
	const Config* cfg;
	const char* objName;

	bool stopParsing(); // returns true if no unused nodes/options
	void markNodeParsed(const std::string& name);
	void markOptionParsed(const std::string& name);
};

class _AstraExport ConfigWriter {
public:
	ConfigWriter(const std::string &name);
	ConfigWriter(const std::string &name, const std::string &type);
	~ConfigWriter();

	Config* getConfig();

	void addInt(const std::string &name, int iValue);
	void addNumerical(const std::string &name, double fValue);
	void addNumericalArray(const std::string &name, const float *pfValues, int iCount);
	void addNumericalMatrix(const std::string &name, const double *pfValues, int iHeight, int iWidth);
	void addID(const std::string &name, int iValue);

	void addOptionNumerical(const std::string &name, double fValue);

private:
	XMLConfig *cfg;
};


} // end namespace

#endif
