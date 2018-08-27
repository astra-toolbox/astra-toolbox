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

#include "PythonPluginAlgorithm.h"

#include "astra/Logging.h"
#include "astra/Utilities.h"
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

PyObject *CPluginAlgorithm::getInstance() const {
	if (instance)
		Py_INCREF(instance);
	return instance;
}

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
        StringUtil::splitString(rows, str, ";");
        PyObject *mat = PyList_New(rows.size());
        for(unsigned int i=0; i<rows.size(); i++){
            StringUtil::splitString(row, rows[i], ",");
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
        StringUtil::splitString(vec, str, ",");
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
