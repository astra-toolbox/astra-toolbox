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

#ifndef _INC_ASTRA_DATA2D
#define _INC_ASTRA_DATA2D

#include "Globals.h"
#include "Data.h"
#include "VolumeGeometry2D.h"
#include "ProjectionGeometry2D.h"

#include <array>
#include <memory>

namespace astra {

class _AstraExport CData2D : public CData<2> {
public:
	// These shape aliases can be less ambiguous than numbered indices with
	// different conventions in C++, Python, Matlab
	int getWidth() const { return m_iDims[0]; }
	int getHeight() const { return m_iDims[1]; }

	std::string description() const;

	// TODO: Consider moving these operations elsewhere
	CData2D& operator+=(const CData2D& _other);
	CData2D& operator*=(const CData2D& _other);
	void copyData(const CData2D& _other);

	// TODO: Improve this set of scalar operations for templated data types
	CData2D& operator*=(float32 _value);
	void setData(float32 _value);
	void clampMin(float32 _value);
	void clampMax(float32 _value);

protected:
	CData2D(int x, int y, CDataStorage *storage) : CData({x, y}, storage) { }
};

template <class G>
class _AstraExport CData2DObject : public CData2D {
protected:
	std::unique_ptr<G> m_pGeometry;
	CData2DObject(int x, int y, std::unique_ptr<G>&& geom, CDataStorage *storage) : CData2D(x, y, storage), m_pGeometry(std::move(geom)) { }
	CData2DObject(int x, int y, const G &geom, CDataStorage *storage) : CData2D(x, y, storage), m_pGeometry(geom.clone()) { }
	virtual ~CData2DObject() { }

public:

	const G& getGeometry() const { return *m_pGeometry; }
	void changeGeometry(std::unique_ptr<G> &&geom) { m_pGeometry = std::move(geom); }
	void changeGeometry(const G &geom) { m_pGeometry.reset(geom.clone()); }
};

class _AstraExport CFloat32ProjectionData2D : public CData2DObject<CProjectionGeometry2D> {
public:

	CFloat32ProjectionData2D(std::unique_ptr<CProjectionGeometry2D>&& geom, CDataStorage *storage) : CData2DObject<CProjectionGeometry2D>(geom->getDetectorCount(), geom->getProjectionAngleCount(), std::move(geom), storage) { }
	CFloat32ProjectionData2D(const CProjectionGeometry2D &geom, CDataStorage *storage) : CData2DObject<CProjectionGeometry2D>(geom.getDetectorCount(), geom.getProjectionAngleCount(), geom, storage) { }

	int getDetectorCount() const { return m_iDims[0]; }
	int getAngleCount() const { return m_iDims[1]; }

	virtual EDataType getType() const { return PROJECTION; }
};

class _AstraExport CFloat32VolumeData2D : public CData2DObject<CVolumeGeometry2D> {
public:
	CFloat32VolumeData2D(std::unique_ptr<CVolumeGeometry2D>&& geom, CDataStorage *storage) : CData2DObject<CVolumeGeometry2D>(geom->getGridColCount(), geom->getGridRowCount(), std::move(geom), storage) { }
	CFloat32VolumeData2D(const CVolumeGeometry2D &geom, CDataStorage *storage) : CData2DObject<CVolumeGeometry2D>(geom.getGridColCount(), geom.getGridRowCount(), geom, storage) { }

	int getColCount() const { return m_iDims[0]; }
	int getRowCount() const { return m_iDims[1]; }

	virtual EDataType getType() const { return VOLUME; }
};


// Utility functions that create CDataMemory and Data2D objects together
_AstraExport CFloat32ProjectionData2D *createCFloat32ProjectionData2DMemory(const CProjectionGeometry2D &geom);
_AstraExport CFloat32ProjectionData2D *createCFloat32ProjectionData2DMemory(std::unique_ptr<CProjectionGeometry2D> &&geom);

_AstraExport CFloat32VolumeData2D *createCFloat32VolumeData2DMemory(const CVolumeGeometry2D &geom);
_AstraExport CFloat32VolumeData2D *createCFloat32VolumeData2DMemory(std::unique_ptr<CVolumeGeometry2D> &&geom);

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32DATA2D
