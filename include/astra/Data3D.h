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

#ifndef _INC_ASTRA_DATA3D
#define _INC_ASTRA_DATA3D

#include "Globals.h"
#include "VolumeGeometry3D.h"
#include "ProjectionGeometry3D.h"
#include "cuda/3d/mem3d.h"

#include <array>

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

	CDataMemory(size_t size) : m_pfData(nullptr) { _allocateData(size); }
	virtual ~CDataMemory() { _freeData(); }

	T* getData() { return m_pfData; }
	const T* getData() const { return m_pfData; }

	virtual bool isMemory() const { return true; }
	virtual bool isGPU() const { return false; }
	virtual bool isFloat32() const { return std::is_same_v<T, float32>; }

protected:
	T* m_pfData;
	CDataMemory() : m_pfData(nullptr) { }

private:
	void _allocateData(size_t size);
	void _freeData();
};



class _AstraExport CData3D {
public:

	typedef enum {BASE, PROJECTION, VOLUME} EDataType;

	virtual ~CData3D() { delete m_storage; }

	int getDimensionCount() const { return 3; }
	std::array<int, 3> getShape() const { return m_iDims; }

	int getWidth() const { return m_iDims[0]; }
	int getHeight() const { return m_iDims[1]; }
	int getDepth() const { return m_iDims[2]; }

	size_t getSize() const { return m_iSize; }

	virtual EDataType getType() const =0;
	std::string description() const;

	CDataStorage *getStorage() const { return m_storage; }

	// Convenience functions as this is the common case
	bool isFloat32Memory() const { return m_storage->isMemory() && m_storage->isFloat32(); }
	float32 *getFloat32Memory() const { return isFloat32Memory() ? dynamic_cast<CDataMemory<float32>*>(m_storage)->getData() : nullptr; }

	bool isInitialized() const { return true; }

protected:
	CData3D(int x, int y, int z, CDataStorage *storage) : m_iDims{x, y, z}, m_iSize((size_t)x*y*z), m_storage(storage) { }


	std::array<int, 3> m_iDims;			///< dimensions of the data (width, height, depth)
	size_t m_iSize;			///< size of the data (width*height*depth)

	CDataStorage *m_storage;
};

template <class G>
class _AstraExport CData3DObject : public CData3D {
protected:
	std::unique_ptr<G> m_pGeometry;
	CData3DObject(int x, int y, int z, std::unique_ptr<G>&& geom, CDataStorage *storage) : CData3D(x, y, z, storage), m_pGeometry(std::move(geom)) { }
	CData3DObject(int x, int y, int z, const G &geom, CDataStorage *storage) : CData3D(x, y, z, storage), m_pGeometry(geom.clone()) { }
	virtual ~CData3DObject() { }

public:

	const G& getGeometry() const { return *m_pGeometry; }
	void changeGeometry(std::unique_ptr<G> &&geom) { m_pGeometry = std::move(geom); }
	void changeGeometry(const G &geom) { m_pGeometry.reset(geom.clone()); }
};

class _AstraExport CFloat32ProjectionData3D : public CData3DObject<CProjectionGeometry3D> {
public:

	CFloat32ProjectionData3D(std::unique_ptr<CProjectionGeometry3D>&& geom, CDataStorage *storage) : CData3DObject<CProjectionGeometry3D>(geom->getDetectorColCount(), geom->getProjectionCount(), geom->getDetectorRowCount(), std::move(geom), storage) { }
	CFloat32ProjectionData3D(const CProjectionGeometry3D &geom, CDataStorage *storage) : CData3DObject<CProjectionGeometry3D>(geom.getDetectorColCount(), geom.getProjectionCount(), geom.getDetectorRowCount(), geom, storage) { }

	int getDetectorRowCount() const { return m_iDims[2]; }
	int getDetectorColCount() const { return m_iDims[0]; }
	int getDetectorTotCount() const { return m_iDims[0] * m_iDims[2]; }
	int getAngleCount() const { return m_iDims[1]; }

	virtual EDataType getType() const { return PROJECTION; }
};

class _AstraExport CFloat32VolumeData3D : public CData3DObject<CVolumeGeometry3D> {
public:
	CFloat32VolumeData3D(std::unique_ptr<CVolumeGeometry3D>&& geom, CDataStorage *storage) : CData3DObject<CVolumeGeometry3D>(geom->getGridColCount(), geom->getGridRowCount(), geom->getGridSliceCount(), std::move(geom), storage) { }
	CFloat32VolumeData3D(const CVolumeGeometry3D &geom, CDataStorage *storage) : CData3DObject<CVolumeGeometry3D>(geom.getGridColCount(), geom.getGridRowCount(), geom.getGridSliceCount(), geom, storage) { }

	int getColCount() const { return m_iDims[0]; }
	int getRowCount() const { return m_iDims[1]; }
	int getSliceCount() const { return m_iDims[2]; }

	virtual EDataType getType() const { return VOLUME; }
};


#ifdef ASTRA_CUDA

class _AstraExport CDataGPU : public CDataStorage {

protected:
	/** Handle for the memory block */
	astraCUDA3d::MemHandle3D m_hnd;
	CDataGPU() { }

public:

	CDataGPU(astraCUDA3d::MemHandle3D hnd) : m_hnd(hnd) { }

	virtual bool isMemory() const { return false; }
	virtual bool isGPU() const { return true; }
	virtual bool isFloat32() const { return true; } // TODO

	astraCUDA3d::MemHandle3D& getHandle() { return m_hnd; }

};

#endif

template class CDataMemory<float32>;
template class CData3DObject<CProjectionGeometry3D>;
template class CData3DObject<CVolumeGeometry3D>;

// Utility functions that create CDataMemory and Data3D objects together
_AstraExport CFloat32ProjectionData3D *createCFloat32ProjectionData3DMemory(const CProjectionGeometry3D &geom);
_AstraExport CFloat32ProjectionData3D *createCFloat32ProjectionData3DMemory(std::unique_ptr<CProjectionGeometry3D> &&geom);

_AstraExport CFloat32VolumeData3D *createCFloat32VolumeData3DMemory(const CVolumeGeometry3D &geom);
_AstraExport CFloat32VolumeData3D *createCFloat32VolumeData3DMemory(std::unique_ptr<CVolumeGeometry3D> &&geom);

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32DATA2D
