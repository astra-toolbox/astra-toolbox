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

#ifndef _INC_ASTRA_PLUGINALGORITHM
#define _INC_ASTRA_PLUGINALGORITHM

#include "astra/Globals.h"

#include <map>
#include <string>

namespace astra {

class CAlgorithm;

class _AstraExport CPluginAlgorithmFactory {

public:
    CPluginAlgorithmFactory() { }
    virtual ~CPluginAlgorithmFactory() { }

    virtual CAlgorithm * getPlugin(const std::string &name) = 0;

    virtual bool registerPlugin(std::string name, std::string className) = 0;
    virtual bool registerPlugin(std::string className) = 0;

    virtual std::map<std::string, std::string> getRegisteredMap() = 0;
    
    virtual std::string getHelp(const std::string &name) = 0;

    static void registerFactory(CPluginAlgorithmFactory *factory) { m_factory = factory; }
	static CPluginAlgorithmFactory* getFactory() { return m_factory; }

private:
    static CPluginAlgorithmFactory *m_factory;
};

}

#endif
