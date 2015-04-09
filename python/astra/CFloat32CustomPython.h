class CFloat32CustomPython : public astra::CFloat32CustomMemory {
public:
    CFloat32CustomPython(PyObject * arrIn)
    {
        arr = arrIn;
        // Set pointer to numpy data pointer
        m_fPtr = (float *)PyArray_DATA(arr);
        // Increase reference count since ASTRA has a reference
        Py_INCREF(arr);
    }
    virtual ~CFloat32CustomPython() {
        // Decrease reference count since ASTRA object is destroyed
        Py_DECREF(arr);
    }
private:
    PyObject* arr;
};