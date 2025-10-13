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
#include "Data.h"
#include "VolumeGeometry3D.h"
#include "ProjectionGeometry3D.h"
#include "cuda/3d/mem3d.h"

#include <array>

namespace astra {

class _AstraExport CData3D : public CData<3> {
public:
	// These shape aliases can be less ambiguous than numbered indices with
	// different conventions in C++, Python, Matlab
	int getWidth() const { return m_iDims[0]; }
	int getHeight() const { return m_iDims[1]; }
	int getDepth() const { return m_iDims[2]; }

	std::string description() const;

protected:
	CData3D(int x, int y, int z, CDataStorage *storage) : CData({x, y, z}, storage) { }
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

// Utility functions that create CDataMemory and Data3D objects together
_AstraExport CFloat32ProjectionData3D *createCFloat32ProjectionData3DMemory(const CProjectionGeometry3D &geom);
_AstraExport CFloat32ProjectionData3D *createCFloat32ProjectionData3DMemory(std::unique_ptr<CProjectionGeometry3D> &&geom);

_AstraExport CFloat32VolumeData3D *createCFloat32VolumeData3DMemory(const CVolumeGeometry3D &geom);
_AstraExport CFloat32VolumeData3D *createCFloat32VolumeData3DMemory(std::unique_ptr<CVolumeGeometry3D> &&geom);

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32DATA2D
