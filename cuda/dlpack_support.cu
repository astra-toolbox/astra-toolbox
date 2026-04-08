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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include "astra/cuda/dlpack_support.h"
#include "astra/cuda/3d/mem3d_internal.h"

#include "astra/Data3D.h"

namespace astraCUDA {

template<class DLT>
class CDataStorageDLPackGPU : public CDataGPU {
public:
	CDataStorageDLPackGPU(DLT* tensor);
	virtual ~CDataStorageDLPackGPU();

protected:
	DLT *m_pTensor;
};

template<class DLT>
CDataStorageDLPackGPU<DLT>::CDataStorageDLPackGPU(DLT *tensor_m)
        : m_pTensor(tensor_m)
{
	DLTensor *tensor = &m_pTensor->dl_tensor;

	uint8_t* data = static_cast<uint8_t*>(tensor->data);
	data += tensor->byte_offset;
	unsigned int pitch = tensor->shape[2];

	ptr.ptr = data;
	ptr.xsize = sizeof(float) * tensor->shape[2];
	ptr.pitch = sizeof(float) * pitch;
	ptr.ysize = tensor->shape[1];
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




astra::CDataStorage *wrapDLTensor(DLManagedTensorVersioned *tensor_m)
{
	return new CDataStorageDLPackGPU<DLManagedTensorVersioned>(tensor_m);
}

astra::CDataStorage *wrapDLTensor(DLManagedTensor *tensor_m)
{
	return new CDataStorageDLPackGPU<DLManagedTensor>(tensor_m);
}

bool isSupportedDLPackGPUType(DLDeviceType type)
{
#ifdef ASTRA_BUILDING_CUDA
	if (type == kDLCUDA || type == kDLCUDAManaged)
		return true;
#endif
#ifdef ASTRA_BUILDING_HIP
	if (type == kDLROCM)
		return true;
#endif
	return false;
}

}
