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

#ifndef ASTRA_PYTHON_SRC_FLOAT32CUSTOMPYTHON_H
#define ASTRA_PYTHON_SRC_FLOAT32CUSTOMPYTHON_H

class GILHelper {
public:
    GILHelper() {
        state = PyGILState_Ensure();
    }
    ~GILHelper() {
        PyGILState_Release(state);
    }
private:
    PyGILState_STATE state;
};

class CFloat32CustomPython : public astra::CFloat32CustomMemory {
public:
    CFloat32CustomPython(PyArrayObject * arrIn)
    {
        GILHelper gil; // hold GIL during this function

        arr = arrIn;
        // Set pointer to numpy data pointer
        m_fPtr = (float *)PyArray_DATA(arr);
        // Increase reference count since ASTRA has a reference
        Py_INCREF(arr);
    }
    virtual ~CFloat32CustomPython() {
        GILHelper gil; // hold GIL during this function

        // Decrease reference count since ASTRA object is destroyed
        Py_DECREF(arr);
    }
private:
    PyArrayObject* arr;
};

#endif
