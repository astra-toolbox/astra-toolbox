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

#ifndef ASTRA_CUDA_MEM3D_INTERNAL_H
#define ASTRA_CUDA_MEM3D_INTERNAL_H

#include "astra/Data.h"
#include "astra/cuda/3d/mem3d.h"

namespace astraCUDA {


// TODO: Turn CDataGPU wrapping cudaArray into separate type
class _AstraExport CDataGPU : public astra::CDataStorage {

protected:
	cudaPitchedPtr ptr;
	cudaArray *arr;

	CDataGPU() : arr(nullptr) { ptr.ptr = nullptr; }

public:

	CDataGPU(cudaPitchedPtr _ptr) : ptr(_ptr), arr(nullptr)  { }
	CDataGPU(cudaArray *_arr) : arr(_arr)  { ptr.ptr = nullptr; }

	virtual bool isMemory() const { return false; }
	virtual bool isGPU() const { return true; }
	virtual bool isFloat32() const { return true; } // TODO

	// The CUDA API isn't very const-friendly, so these functions drop
	// consts from the pointers.
	cudaArray* getArray() const { return const_cast<cudaArray*>(arr); }
	cudaPitchedPtr getPtr() const { return ptr; }

};

}

#endif
