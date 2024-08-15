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

#ifndef _INC_ASTRA_XMLCONFIG
#define _INC_ASTRA_XMLCONFIG

#include "Globals.h"
#include "XMLNode.h"
#include "XMLDocument.h"
#include "Config.h"

namespace astra {

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
