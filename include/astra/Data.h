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

#ifndef _INC_ASTRA_DATA
#define _INC_ASTRA_DATA

#include "Globals.h"

#include <array>
#include <memory>
#include <numeric>
#include <functional>

namespace astra {

class _AstraExport CDataStorage {
public:
	CDataStorage() { }
	virtual ~CDataStorage() { }

	virtual bool isMemory() const =0;
	virtual bool isGPU() const =0;
	virtual bool isFloat32() const =0;
};

template <typename T>
class _AstraExport CDataMemory : public CDataStorage {
public:

	CDataMemory(size_t size) : m_bOwnData(true), m_pfData(nullptr) { _allocateData(size); }
	virtual ~CDataMemory() { _freeData(); }

	T* getData() { return m_pfData; }
	const T* getData() const { return m_pfData; }

	virtual bool isMemory() const { return true; }
	virtual bool isGPU() const { return false; }
	virtual bool isFloat32() const { return std::is_same_v<T, float32>; }

protected:
	bool m_bOwnData;
	T* m_pfData;
	CDataMemory() : m_bOwnData(false), m_pfData(nullptr) { }

private:
	void _allocateData(size_t size);
	void _freeData();
};


// TODO: Consider a common base class
// Consider functions checking if two object have same dimensions and/or geom
// Clean up arithmetic operations (in Data2D)

template <size_t D>
class _AstraExport CData {
public:
	typedef enum {BASE, PROJECTION, VOLUME} EDataType;

	virtual ~CData() { delete m_storage; }

	int getDimensionCount() const { return D; }
	std::array<int, D> getShape() const { return m_iDims; }

	size_t getSize() const { return m_iSize; }

	virtual EDataType getType() const =0;

	CDataStorage *getStorage() { return m_storage; }
	const CDataStorage *getStorage() const { return m_storage; }

	// Convenience functions as this is the common case
	bool isFloat32Memory() const { return m_storage->isMemory() && m_storage->isFloat32(); }
	float32 *getFloat32Memory() { return isFloat32Memory() ? dynamic_cast<CDataMemory<float32>*>(m_storage)->getData() : nullptr; }
	const float32 *getFloat32Memory() const { return isFloat32Memory() ? dynamic_cast<const CDataMemory<float32>*>(m_storage)->getData() : nullptr; }

	// Legacy function. Maybe remove?
	bool isInitialized() const { return true; }

protected:
	CData(std::array<int, D> dims, CDataStorage *storage)
		: m_iDims(dims), m_storage(storage)
	{
		m_iSize = std::reduce(std::cbegin(m_iDims), std::cend(m_iDims), (size_t)1, std::multiplies<size_t>());
	}

	std::array<int, D> m_iDims;
	size_t m_iSize;

	CDataStorage *m_storage;
};



}

#endif
