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


#include "dlpack/dlpack.h"

#include "astra/Data2D.h"
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

static void getNonSingletonDims(DLTensor *tensor,
                                std::vector<int64_t> &nonsingular_dims,
                                std::vector<int64_t> &nonsingular_strides)
{
	for (int i = 0; i < tensor->ndim; ++i) {
		if (tensor->shape[i] > 1) {
			nonsingular_dims.push_back(tensor->shape[i]);
			nonsingular_strides.push_back(tensor->strides[i]);
		}
	}
}

#ifdef ASTRA_CUDA
template<class DLT>
CDataStorageDLPackGPU<DLT>::CDataStorageDLPackGPU(DLT *tensor_m)
	: m_pTensor(tensor_m)
{
	DLTensor *tensor = &m_pTensor->dl_tensor;

	uint8_t* data = static_cast<uint8_t*>(tensor->data);
	data += tensor->byte_offset;

	unsigned int pitch = tensor->shape[2];
	if (tensor->strides) {
		std::vector<int64_t> non_singleton_dims, non_singleton_strides;
		getNonSingletonDims(tensor, non_singleton_dims, non_singleton_strides);
		if (non_singleton_dims.size() > 1)
			// Input potentially non-contiguous, use the provided stride
			pitch = non_singleton_strides[non_singleton_strides.size() - 2];
		else if (non_singleton_dims.size() == 1)
			// No meaningful stride, but need to ensure pitch is at least as large as the input size
			pitch = non_singleton_dims[0];
	}

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


static bool isContiguous(DLTensor *tensor, bool allowPitch)
{
	if (!tensor->strides)
		return true;

	// Ignore singleton dimensions as they don't affect contiguity and tensor producers
	// may optionally set their strides to 1
	std::vector<int64_t> non_singleton_dims, non_singleton_strides;
	getNonSingletonDims(tensor, non_singleton_dims, non_singleton_strides);

	if (non_singleton_dims.size() < 2)
		return true;

	int64_t accumulator = 1;
	for (int i = non_singleton_dims.size() - 1; i >= 0; --i) {
		if (non_singleton_strides[i] != accumulator)
			return false;
		if (allowPitch && i == non_singleton_dims.size()-1)
			// Accept non-contiguous second-to-last non-singular dimension,
			// since such data can be represented using cudaPitchedPtr
			accumulator *= non_singleton_strides[i-1];
		else
			accumulator *= non_singleton_dims[i];
	}
	return true;
}


template<size_t D>
bool checkDLTensor(DLTensor *tensor, std::array<int, D> dims, bool allowPitch, std::string &error)
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
	if constexpr (D == 2) {
		if (tensor->ndim != 2) {
			error = "Data must be two-dimensional";
			return false;
		}

		if (tensor->shape[0] != dims[1] || tensor->shape[1] != dims[0])
		{
			error = astra::StringUtil::format("Data shape (%zd x %zd) does not match geometry (%d x %d)", tensor->shape[0], tensor->shape[1], dims[1], dims[0]);
			return false;
		}
	}
	if constexpr (D == 3) {
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
	}

	if (!isContiguous(tensor, allowPitch)) {
		error = "Data must be contiguous";
		return false;
	}

	error = "";
	return true;
}


template<class DLT, size_t D>
astra::CDataStorage *getDLTensorStorage(DLT *tensor_m, std::array<int, D> dims, std::string &error)
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
	// TODO: Add support for kDLROCM, for the case when astra is built with
	// cuda-but-actually-hip, and later when it is in its own namespace
	default:
		error = "Unsupported dlpack device type";
		return nullptr;
	}
}

template<size_t D>
astra::CDataStorage *getDLTensorStorage(PyObject *obj, std::array<int, D> dims, std::string &error)
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

		if (tensor_m->flags & DLPACK_FLAG_BITMASK_READ_ONLY) {
			error = "Read-only dlpack tensor was provided";
			return nullptr;
		}

		// NB: We ignore the DLPACK_FLAG_BITMASK_IS_COPIED flag.
		// If we ever add explicit support for differentiating between
		// input and output objects, we may want to return an error
		// when using a copied tensor for output.

		storage = getDLTensorStorage(tensor_m, dims, error);
		if (storage) {
			// All checks passed, so we can officially consume this dltensor
			PyCapsule_SetName(obj, "used_dltensor_versioned");
		}
	}

	return storage;
}

astra::CFloat32VolumeData2D* getDLTensor(PyObject *obj, const astra::CVolumeGeometry2D &pGeom, std::string &error)
{
	if (!PyCapsule_CheckExact(obj))
		return nullptr;

	// x,y,z
	std::array<int, 2> dims{pGeom.getGridColCount(), pGeom.getGridRowCount()};

	astra::CDataStorage *storage = getDLTensorStorage(obj, dims, error);
	if (!storage)
		return nullptr;

	return new astra::CFloat32VolumeData2D(pGeom, storage);
}
astra::CFloat32ProjectionData2D* getDLTensor(PyObject *obj, const astra::CProjectionGeometry2D &pGeom, std::string &error)
{
	if (!PyCapsule_CheckExact(obj))
		return nullptr;

	// x,y,z
	std::array<int, 2> dims{pGeom.getDetectorCount(), pGeom.getProjectionAngleCount()};

	astra::CDataStorage *storage = getDLTensorStorage(obj, dims, error);
	if (!storage)
		return nullptr;

	return new astra::CFloat32ProjectionData2D(pGeom, storage);
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
