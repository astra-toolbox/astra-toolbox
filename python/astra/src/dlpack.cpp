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


// TODO: fix include paths
#include "../../lib/include/dlpack/dlpack.h"

#include "astra/Data3D.h"

#include <Python.h>
#include <cstdio>

template<class DLT>
class CDataStorageDLPackCPU : public astra::CDataMemory<float> {
public:
	CDataStorageDLPackCPU(DLT *tensor);
	virtual ~CDataStorageDLPackCPU();

protected:
	DLT *m_pTensor;

};

#ifdef ASTRA_CUDA
template<class DLT>
class CDataStorageDLPackGPU : public astra::CDataGPU {
public:
	CDataStorageDLPackGPU(DLT* tensor);
	virtual ~CDataStorageDLPackGPU();

protected:
	DLT *m_pTensor;
};
#endif

template<class DLT>
CDataStorageDLPackCPU<DLT>::CDataStorageDLPackCPU(DLT *tensor_m)
	: m_pTensor(tensor_m)
{
	// We assume all sanity checks have already been done

	uint8_t* data = static_cast<uint8_t*>(m_pTensor->dl_tensor.data);
	data += m_pTensor->dl_tensor.byte_offset;

	this->m_pfData = reinterpret_cast<float*>(data);
}

template<class DLT>
CDataStorageDLPackCPU<DLT>::~CDataStorageDLPackCPU()
{
	if (m_pTensor) {
		assert(m_pTensor->deleter);
		m_pTensor->deleter(m_pTensor);
	}
	m_pTensor = nullptr;

	// Prevent the parent destructor from deleting this memory
	m_pfData = nullptr;
}

#ifdef ASTRA_CUDA
template<class DLT>
CDataStorageDLPackGPU<DLT>::CDataStorageDLPackGPU(DLT *tensor_m)
	: m_pTensor(tensor_m)
{
	DLTensor *tensor = &m_pTensor->dl_tensor;

	uint8_t* data = static_cast<uint8_t*>(tensor->data);
	data += tensor->byte_offset;

	unsigned int pitch = tensor->shape[0];
	if (tensor->strides)
		pitch = tensor->strides[1];

	m_hnd = astraCUDA3d::wrapHandle(reinterpret_cast<float*>(data), tensor->shape[2], tensor->shape[1], tensor->shape[0], pitch);
}


template<class DLT>
CDataStorageDLPackGPU<DLT>::~CDataStorageDLPackGPU()
{
	if (m_pTensor) {
		assert(m_pTensor->deleter);
		m_pTensor->deleter(m_pTensor);
	}
	m_pTensor = nullptr;
}
#endif


bool checkDLTensor(DLTensor *tensor, std::array<int, 3> dims, bool allowPitch, std::string &error)
{
	// data type
	if (tensor->dtype.code != kDLFloat || tensor->dtype.bits != 32)
	{
		error = "Data must be float32";
		return false;
	}
	if (tensor->dtype.lanes != 1) {
		error = "Data must be single-channel";
		return false;
	}

	// shape
	if (tensor->ndim != 3) {
		error = "Data must be three-dimensional";
		return false;
	}

	if (tensor->shape[0] != dims[2] || tensor->shape[1] != dims[1] ||
	    tensor->shape[2] != dims[0])
	{
		error = astra::StringUtil::format("Data shape (%zd x %zd x %zd) does not match geometry (%d x %d x %d)", tensor->shape[0], tensor->shape[1], tensor->shape[2], dims[2], dims[1], dims[0]);
		return false;
	}

	if (tensor->strides) {
		int64_t acc = 1;
		for (int i = tensor->ndim-1; i >= 0; --i) {
			// We don't check this for dimensions where the shape is 1.
			// There the stride is not relevant, and torch can set it to 1.
			if (tensor->shape[i] >= 2 && tensor->strides[i] != acc) {
				if (allowPitch)
					error = "Data must be contiguous in all dimensions except first";
				else
					error = "Data must be contiguous";
				return false;
			}
			if (i == tensor->ndim-1 && allowPitch) {
				// allow different stride in x. This stride
				// will be the first valid stride for a lower
				// axis
				int j = i-1;
				while (j >= 0 && tensor->shape[j] < 2)
					--j;
				acc *= tensor->strides[j];
			} else
				acc *= tensor->shape[i];
		}
	}

	error = "";
	return true;
}


template<class DLT>
astra::CDataStorage *getDLTensorStorage(DLT *tensor_m, std::array<int, 3> dims, std::string &error)
{
	DLTensor *tensor = &tensor_m->dl_tensor;

	switch (tensor->device.device_type) {
	case kDLCPU:
	case kDLCUDAHost:
		if (!checkDLTensor(tensor, dims, false, error))
			return nullptr;
		return new CDataStorageDLPackCPU(tensor_m);
#ifdef ASTRA_CUDA
	case kDLCUDA:
	case kDLCUDAManaged:
		if (!checkDLTensor(tensor, dims, true, error))
			return nullptr;
		return new CDataStorageDLPackGPU(tensor_m);
#endif
	default:
		error = "Unsupported dlpack device type";
		return nullptr;
	}
}

astra::CDataStorage *getDLTensorStorage(PyObject *obj, std::array<int, 3> dims, std::string &error)
{
	if (!PyCapsule_CheckExact(obj)) {
		error = "Invalid capsule";
		return nullptr;
	}

	astra::CDataStorage *storage = nullptr;

	if (PyCapsule_IsValid(obj, "dltensor")) {
		// DLPack pre-1.0 unversioned interface
		void *ptr = PyCapsule_GetPointer(obj, "dltensor");
		if (!ptr) {
			error = "Invalid dlpack capsule";
			return nullptr;
		}
		DLManagedTensor *tensor_m = static_cast<DLManagedTensor*>(ptr);
		storage = getDLTensorStorage(tensor_m, dims, error);

		if (storage) {
			// All checks passed, so we can officially consume this dltensor
			PyCapsule_SetName(obj, "used_dltensor");
		}

	} else if (PyCapsule_IsValid(obj, "dltensor_versioned")) {
		// DLPack 1.0 versioned interface
		void *ptr = PyCapsule_GetPointer(obj, "dltensor_versioned");
		if (!ptr) {
			error = "Invalid dlpack capsule";
			return nullptr;
		}
		DLManagedTensorVersioned *tensor_m = static_cast<DLManagedTensorVersioned*>(ptr);

		if (tensor_m->version.major > 1) {
			error = astra::StringUtil::format("Unsupported dlpack version %d.%d", tensor_m->version.major, tensor_m->version.minor);
			return nullptr;
		}

		// TODO: handle read-only and copy flags
		if (tensor_m->flags)
			printf("unhandled flags: %zx\n", tensor_m->flags);

		storage = getDLTensorStorage(tensor_m, dims, error);
		if (storage) {
			// All checks passed, so we can officially consume this dltensor
			PyCapsule_SetName(obj, "used_dltensor_versioned");
		}
	}

	return storage;
}

astra::CFloat32VolumeData3D* getDLTensor(PyObject *obj, const astra::CVolumeGeometry3D &pGeom, std::string &error)
{
	if (!PyCapsule_CheckExact(obj))
		return nullptr;

	// x,y,z
	std::array<int, 3> dims{pGeom.getGridColCount(), pGeom.getGridRowCount(), pGeom.getGridSliceCount()};

	astra::CDataStorage *storage = getDLTensorStorage(obj, dims, error);
	if (!storage)
		return nullptr;

	return new astra::CFloat32VolumeData3D(pGeom, storage);
}
astra::CFloat32ProjectionData3D* getDLTensor(PyObject *obj, const astra::CProjectionGeometry3D &pGeom, std::string &error)
{
	if (!PyCapsule_CheckExact(obj))
		return nullptr;

	// x,y,z
	std::array<int, 3> dims{pGeom.getDetectorColCount(), pGeom.getProjectionCount(), pGeom.getDetectorRowCount()};

	astra::CDataStorage *storage = getDLTensorStorage(obj, dims, error);
	if (!storage)
		return nullptr;

	return new astra::CFloat32ProjectionData3D(pGeom, storage);
}
