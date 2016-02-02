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

#ifdef ASTRA_PYTHON

#include "PythonPluginAlgorithm.h"

#include "astra/Logging.h"
#include "astra/Utilities.h"
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <Python.h>
#include "bytesobject.h"

namespace astra {



void logPythonError(){
    if(PyErr_Occurred()){
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *traceback = PyImport_ImportModule("traceback");
        if(traceback!=NULL){
            PyObject *exc;
            if(ptraceback==NULL){
                exc = PyObject_CallMethod(traceback,"format_exception_only","OO",ptype, pvalue);
            }else{
                exc = PyObject_CallMethod(traceback,"format_exception","OOO",ptype, pvalue, ptraceback);
            }
            if(exc!=NULL){
                PyObject *six = PyImport_ImportModule("six");
                if(six!=NULL){
                    PyObject *iter = PyObject_GetIter(exc);
                    if(iter!=NULL){
                        PyObject *line;
                        std::string errStr = "";
                        while(line = PyIter_Next(iter)){
                            PyObject *retb = PyObject_CallMethod(six,"b","O",line);
                            if(retb!=NULL){
                                errStr += std::string(PyBytes_AsString(retb));
                                Py_DECREF(retb);
                            }
                            Py_DECREF(line);
                        }
                        ASTRA_ERROR("%s",errStr.c_str());
                        Py_DECREF(iter);
                    }
                    Py_DECREF(six);
                }
                Py_DECREF(exc);
            }
            Py_DECREF(traceback);
        }
        if(ptype!=NULL) Py_DECREF(ptype);
        if(pvalue!=NULL) Py_DECREF(pvalue);
        if(ptraceback!=NULL) Py_DECREF(ptraceback);
    }
}


CPluginAlgorithm::CPluginAlgorithm(PyObject* pyclass){
    instance = PyObject_CallObject(pyclass, NULL);
    if(instance==NULL) logPythonError();
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
    if(retVal==NULL){
        logPythonError();
        return false;
    }
    m_bIsInitialized = true;
    Py_DECREF(retVal);
    return m_bIsInitialized;
}

void CPluginAlgorithm::run(int _iNrIterations){
    if(instance==NULL) return;
    PyGILState_STATE state = PyGILState_Ensure();
    PyObject *retVal = PyObject_CallMethod(instance, "run", "i",_iNrIterations);
    if(retVal==NULL){
        logPythonError();
    }else{
        Py_DECREF(retVal);
    }
    PyGILState_Release(state);
}

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
    boost::split(items, str, boost::is_any_of("."));
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
                PyList_SetItem(rowlist, j, PyFloat_FromDouble(StringUtil::stringToDouble(row[j])));
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
            PyList_SetItem(veclist, i, PyFloat_FromDouble(StringUtil::stringToDouble(vec[i])));
        }
        return veclist;
    }
    try{
        return PyLong_FromLong(StringUtil::stringToInt(str));
    }catch(const StringUtil::bad_cast &){
        try{
            return PyFloat_FromDouble(StringUtil::stringToDouble(str));
        }catch(const StringUtil::bad_cast &){
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
            PyObject *obj;
            if(subnode.hasAttribute("value")){
                obj = stringToPythonValue(subnode.getAttribute("value"));
            }else{
                obj = stringToPythonValue(subnode.getContent());
            }
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
