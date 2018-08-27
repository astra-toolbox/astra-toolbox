/*
-----------------------------------------------------------------------
Copyright: 2010-2016, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#ifdef ASTRA_PYTHON

#include "PythonPluginAlgorithmFactory.h"
#include "PythonPluginAlgorithm.h"

#include "astra/Logging.h"
#include "astra/Utilities.h"
#include <iostream>
#include <fstream>
#include <string>

#include <Python.h>
#include "bytesobject.h"

namespace astra {

void logPythonError();

CPythonPluginAlgorithmFactory::CPythonPluginAlgorithmFactory(){
    if(!Py_IsInitialized()){
        Py_Initialize();
        PyEval_InitThreads();
    }
    pluginDict = PyDict_New();
    inspect = PyImport_ImportModule("inspect");
    six = PyImport_ImportModule("six");
}

CPythonPluginAlgorithmFactory::~CPythonPluginAlgorithmFactory(){
    if(pluginDict!=NULL){
        Py_DECREF(pluginDict);
    }
    if(inspect!=NULL) Py_DECREF(inspect);
    if(six!=NULL) Py_DECREF(six);
}

PyObject * getClassFromString(std::string str){
    std::vector<std::string> items;
    StringUtil::splitString(items, str, ".");
    PyObject *pyclass = PyImport_ImportModule(items[0].c_str());
    if(pyclass==NULL){
        logPythonError();
        return NULL;
    }
    PyObject *submod = pyclass;
    for(unsigned int i=1;i<items.size();i++){
        submod = PyObject_GetAttrString(submod,items[i].c_str());
        Py_DECREF(pyclass);
        pyclass = submod;
        if(pyclass==NULL){
            logPythonError();
            return NULL;
        }
    }
    return pyclass;
}

bool CPythonPluginAlgorithmFactory::registerPlugin(std::string name, std::string className){
    PyObject *str = PyBytes_FromString(className.c_str());
    PyDict_SetItemString(pluginDict, name.c_str(), str);
    Py_DECREF(str);
    return true;
}

bool CPythonPluginAlgorithmFactory::registerPlugin(std::string className){
    PyObject *pyclass = getClassFromString(className);
    if(pyclass==NULL) return false;
    bool ret = registerPluginClass(pyclass);
    Py_DECREF(pyclass);
    return ret;
}

bool CPythonPluginAlgorithmFactory::registerPluginClass(std::string name, PyObject * className){
    PyDict_SetItemString(pluginDict, name.c_str(), className);
    return true;
}

bool CPythonPluginAlgorithmFactory::registerPluginClass(PyObject * className){
    PyObject *astra_name = PyObject_GetAttrString(className,"astra_name");
    if(astra_name==NULL){
        logPythonError();
        return false;
    }
    PyObject *retb = PyObject_CallMethod(six,"b","O",astra_name);
    if(retb!=NULL){
        PyDict_SetItemString(pluginDict,PyBytes_AsString(retb),className);
        Py_DECREF(retb);
    }else{
        logPythonError();
    }
    Py_DECREF(astra_name);
    return true;
}

CAlgorithm * CPythonPluginAlgorithmFactory::getPlugin(const std::string &name){
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

PyObject * CPythonPluginAlgorithmFactory::getRegistered(){
    Py_INCREF(pluginDict);
    return pluginDict;
}

std::map<std::string, std::string> CPythonPluginAlgorithmFactory::getRegisteredMap(){
    std::map<std::string, std::string> ret;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(pluginDict, &pos, &key, &value)) {
        PyObject *keystr = PyObject_Str(key);
        if(keystr!=NULL){
            PyObject *valstr = PyObject_Str(value);
            if(valstr!=NULL){
                PyObject * keyb = PyObject_CallMethod(six,"b","O",keystr);
                if(keyb!=NULL){
                    PyObject * valb = PyObject_CallMethod(six,"b","O",valstr);
                    if(valb!=NULL){
                        ret[PyBytes_AsString(keyb)] = PyBytes_AsString(valb);
                        Py_DECREF(valb);
                    }
                    Py_DECREF(keyb);
                }
                Py_DECREF(valstr);
            }
            Py_DECREF(keystr);
        }
        logPythonError();
    }
    return ret;
}

std::string CPythonPluginAlgorithmFactory::getHelp(const std::string &name){
    PyObject *className = PyDict_GetItemString(pluginDict, name.c_str());
    if(className==NULL){
        ASTRA_ERROR("Plugin %s not found!",name.c_str());
        PyErr_Clear();
        return "";
    }
    std::string ret = "";
    PyObject *pyclass;
    if(PyBytes_Check(className)){
        std::string str = std::string(PyBytes_AsString(className));
        pyclass = getClassFromString(str);
    }else{
        pyclass = className;
    }
    if(pyclass==NULL) return "";
    if(inspect!=NULL && six!=NULL){
        PyObject *retVal = PyObject_CallMethod(inspect,"getdoc","O",pyclass);
        if(retVal!=NULL){
            if(retVal!=Py_None){
                PyObject *retb = PyObject_CallMethod(six,"b","O",retVal);
                if(retb!=NULL){
                    ret = std::string(PyBytes_AsString(retb));
                    Py_DECREF(retb);
                }
            }
            Py_DECREF(retVal);
        }else{
            logPythonError();
        }
    }
    if(PyBytes_Check(className)){
        Py_DECREF(pyclass);
    }
    return ret;
}

DEFINE_SINGLETON(CPythonPluginAlgorithmFactory);

}
#endif
