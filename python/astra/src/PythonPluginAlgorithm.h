/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

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
*/

#ifndef _INC_PYTHONPLUGINALGORITHM
#define _INC_PYTHONPLUGINALGORITHM

#ifdef ASTRA_PYTHON

#include "astra/Algorithm.h"
#include "astra/Singleton.h"
#include "astra/XMLDocument.h"
#include "astra/XMLNode.h"
#include "astra/PluginAlgorithm.h"

#include <Python.h>

namespace astra {
class CPluginAlgorithm : public CAlgorithm {

public:

    CPluginAlgorithm(PyObject* pyclass);
    ~CPluginAlgorithm();

    bool initialize(const Config& _cfg);
    void run(int _iNrIterations);

private:
    PyObject * instance;

};

class CPythonPluginAlgorithmFactory : public CPluginAlgorithmFactory, public Singleton<CPythonPluginAlgorithmFactory> {

public:

    CPythonPluginAlgorithmFactory();
    virtual ~CPythonPluginAlgorithmFactory();

    virtual CAlgorithm * getPlugin(const std::string &name);

    virtual bool registerPlugin(std::string name, std::string className);
    virtual bool registerPlugin(std::string className);
    bool registerPluginClass(std::string name, PyObject * className);
    bool registerPluginClass(PyObject * className);

    PyObject * getRegistered();
    virtual std::map<std::string, std::string> getRegisteredMap();

    virtual std::string getHelp(const std::string &name);

private:
    PyObject * pluginDict;
    PyObject *inspect, *six;
};

PyObject* XMLNode2dict(XMLNode node);

}


#endif

#endif
