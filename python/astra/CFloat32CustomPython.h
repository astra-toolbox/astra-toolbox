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
