class CFloat32CustomPython : public astra::CFloat32CustomMemory {
public:
    CFloat32CustomPython(PyArrayObject * arrIn)
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
    PyArrayObject* arr;
};

template<typename T>
class CDataStoragePython : public astra::CDataMemory<T> {
public:
    CDataStoragePython(PyArrayObject *arrIn)
    {
        arr = arrIn;
        // Set pointer to numpy data pointer
        this->m_pfData = (T *)PyArray_DATA(arr);
        // Increase reference count since ASTRA has a reference
        Py_INCREF(arr);
    }
    virtual ~CDataStoragePython() {
        // Decrease reference count since ASTRA object is destroyed
        Py_DECREF(arr);

        this->m_pfData = nullptr;
    }


private:
    PyArrayObject* arr;
};
