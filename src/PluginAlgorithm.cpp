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

#ifdef ASTRA_PYTHON

#include "astra/PluginAlgorithm.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>
#include <string>

namespace astra {

CPluginAlgorithm::CPluginAlgorithm(PyObject* pyclass){
    instance = PyObject_CallObject(pyclass, NULL);
}

CPluginAlgorithm::~CPluginAlgorithm(){
    if(instance!=NULL){
        Py_DECREF(instance);
        instance = NULL;
    }
}

bool CPluginAlgorithm::initialize(const Config& _cfg){
    if(instance==NULL) return false;
    PyObject *cfgDict = XMLNode2dict(_cfg.self);
    PyObject *retVal = PyObject_CallMethod(instance, "astra_init", "O",cfgDict);
    Py_DECREF(cfgDict);
    if(retVal==NULL) return false;
    m_bIsInitialized = true;
    Py_DECREF(retVal);
    return m_bIsInitialized;
}

void CPluginAlgorithm::run(int _iNrIterations){
    if(instance==NULL) return;
    PyObject *retVal = PyObject_CallMethod(instance, "run", "i",_iNrIterations);
    if(retVal==NULL) return;
    Py_DECREF(retVal);
}

const char ps =
#ifdef _WIN32
                            '\\';
#else
                            '/';
#endif

std::vector<std::string> CPluginAlgorithmFactory::getPluginPathList(){
    std::vector<std::string> list;
    list.push_back("/etc/astra-toolbox");
    PyObject *ret, *retb;
    ret = PyObject_CallMethod(inspect,"getfile","O",astra);
    if(ret!=NULL){
        retb = PyObject_CallMethod(six,"b","O",ret);
        Py_DECREF(ret);
        if(retb!=NULL){
            std::string astra_inst (PyBytes_AsString(retb));
            Py_DECREF(retb);
            ret = PyObject_CallMethod(ospath,"dirname","s",astra_inst.c_str());
            if(ret!=NULL){
                retb = PyObject_CallMethod(six,"b","O",ret);
                Py_DECREF(ret);
                if(retb!=NULL){
                    list.push_back(std::string(PyBytes_AsString(retb)));
                    Py_DECREF(retb);
                }
            }
        }
    }
    ret = PyObject_CallMethod(ospath,"expanduser","s","~");
    if(ret!=NULL){
        retb = PyObject_CallMethod(six,"b","O",ret);
        Py_DECREF(ret);
        if(retb!=NULL){
            list.push_back(std::string(PyBytes_AsString(retb)) + ps + ".astra-toolbox");
            Py_DECREF(retb);
        }
    }
    const char *envval = getenv("ASTRA_PLUGIN_PATH");
    if(envval!=NULL){
        list.push_back(std::string(envval));
    }
    return list;
}

CPluginAlgorithmFactory::CPluginAlgorithmFactory(){
    Py_Initialize();
    pluginDict = PyDict_New();
    ospath = PyImport_ImportModule("os.path");
    inspect = PyImport_ImportModule("inspect");
    six = PyImport_ImportModule("six");
    astra = PyImport_ImportModule("astra");
    std::vector<std::string> fls = getPluginPathList();
    std::vector<std::string> items;
    for(unsigned int i=0;i<fls.size();i++){
        std::ifstream fs ((fls[i]+ps+"plugins.txt").c_str());
        if(!fs.is_open()) continue;
        std::string line;
        while (std::getline(fs,line)){
            boost::split(items, line, boost::is_any_of(" "));
            if(items.size()<2) continue;
            PyObject *str = PyBytes_FromString(items[1].c_str());
            PyDict_SetItemString(pluginDict,items[0].c_str(),str);
            Py_DECREF(str);
        }
        fs.close();
    }
}

CPluginAlgorithmFactory::~CPluginAlgorithmFactory(){
    if(pluginDict!=NULL){
        Py_DECREF(pluginDict);
    }
}

bool CPluginAlgorithmFactory::registerPlugin(std::string name, std::string className){
    PyObject *str = PyBytes_FromString(className.c_str());
    PyDict_SetItemString(pluginDict, name.c_str(), str);
    Py_DECREF(str);
    return true;
}

bool CPluginAlgorithmFactory::registerPluginClass(std::string name, PyObject * className){
    PyDict_SetItemString(pluginDict, name.c_str(), className);
    return true;
}

PyObject * getClassFromString(std::string str){
    std::vector<std::string> items;
    boost::split(items, str, boost::is_any_of("."));
    PyObject *pyclass = PyImport_ImportModule(items[0].c_str());
    if(pyclass==NULL) return NULL;
    PyObject *submod = pyclass;
    for(unsigned int i=1;i<items.size();i++){
        submod = PyObject_GetAttrString(submod,items[i].c_str());
        Py_DECREF(pyclass);
        pyclass = submod;
        if(pyclass==NULL) return NULL;
    }
    return pyclass;
}

CPluginAlgorithm * CPluginAlgorithmFactory::getPlugin(std::string name){
    PyObject *className = PyDict_GetItemString(pluginDict, name.c_str());
    if(className==NULL) return NULL;
    CPluginAlgorithm *alg = NULL;
    if(PyBytes_Check(className)){
        std::string str = std::string(PyBytes_AsString(className));
    	PyObject *pyclass = getClassFromString(str);
        if(pyclass!=NULL){
            alg = new CPluginAlgorithm(pyclass);
            Py_DECREF(pyclass);
        }
    }else{
        alg = new CPluginAlgorithm(className);
    }
    return alg;
}

PyObject * CPluginAlgorithmFactory::getRegistered(){
    Py_INCREF(pluginDict);
    return pluginDict;
}

std::string CPluginAlgorithmFactory::getHelp(std::string name){
    PyObject *className = PyDict_GetItemString(pluginDict, name.c_str());
    if(className==NULL) return "";
    std::string str = std::string(PyBytes_AsString(className));
    std::string ret = "";
    PyObject *pyclass = getClassFromString(str);
    if(pyclass==NULL) return "";
    PyObject *module = PyImport_ImportModule("inspect");
    if(module!=NULL){
        PyObject *retVal = PyObject_CallMethod(module,"getdoc","O",pyclass);
        if(retVal!=NULL){
            PyObject *retb = PyObject_CallMethod(six,"b","O",retVal);
            Py_DECREF(retVal);
            if(retVal!=NULL){
                ret = std::string(PyBytes_AsString(retb));
                Py_DECREF(retb);
            }
        }
        Py_DECREF(module);
    }
    Py_DECREF(pyclass);
    return ret;
}

DEFINE_SINGLETON(CPluginAlgorithmFactory);

#if PY_MAJOR_VERSION >= 3
PyObject * pyStringFromString(std::string str){
    return PyUnicode_FromString(str.c_str());
}
#else
PyObject * pyStringFromString(std::string str){
    return PyBytes_FromString(str.c_str());
}
#endif

PyObject* stringToPythonValue(std::string str){
    if(str.find(";")!=std::string::npos){
        std::vector<std::string> rows, row;
        boost::split(rows, str, boost::is_any_of(";"));
        PyObject *mat = PyList_New(rows.size());
        for(unsigned int i=0; i<rows.size(); i++){
            boost::split(row, rows[i], boost::is_any_of(","));
            PyObject *rowlist = PyList_New(row.size());
            for(unsigned int j=0;j<row.size();j++){
                PyList_SetItem(rowlist, j, PyFloat_FromDouble(boost::lexical_cast<double>(row[j])));
            }
            PyList_SetItem(mat, i, rowlist);
        }
        return mat;
    }
    if(str.find(",")!=std::string::npos){
        std::vector<std::string> vec;
        boost::split(vec, str, boost::is_any_of(","));
        PyObject *veclist = PyList_New(vec.size());
        for(unsigned int i=0;i<vec.size();i++){
            PyList_SetItem(veclist, i, PyFloat_FromDouble(boost::lexical_cast<double>(vec[i])));
        }
        return veclist;
    }
    try{
        return PyLong_FromLong(boost::lexical_cast<long>(str));
    }catch(const boost::bad_lexical_cast &){
        try{
            return PyFloat_FromDouble(boost::lexical_cast<double>(str));
        }catch(const boost::bad_lexical_cast &){
            return pyStringFromString(str);
        }
    }
}

PyObject* XMLNode2dict(XMLNode node){
    PyObject *dct = PyDict_New();
    PyObject *opts = PyDict_New();
    if(node.hasAttribute("type")){
        PyObject *obj = pyStringFromString(node.getAttribute("type").c_str());
        PyDict_SetItemString(dct, "type", obj);
        Py_DECREF(obj);
    }
    std::list<XMLNode> nodes = node.getNodes();
    std::list<XMLNode>::iterator it = nodes.begin();
    while(it!=nodes.end()){
        XMLNode subnode = *it;
        if(subnode.getName()=="Option"){
            PyObject *obj = stringToPythonValue(subnode.getAttribute("value"));
            PyDict_SetItemString(opts, subnode.getAttribute("key").c_str(), obj);
            Py_DECREF(obj);
        }else{
            PyObject *obj = stringToPythonValue(subnode.getContent());
            PyDict_SetItemString(dct, subnode.getName().c_str(), obj);
            Py_DECREF(obj);
        }
        ++it;
    }
    PyDict_SetItemString(dct, "options", opts);
    Py_DECREF(opts);
    return dct;
}

}
#endif