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

#include <sstream>

#include "astra/Data3D.h"

namespace astra {

std::string CData3D::description() const
{
	std::stringstream res;
	res << m_iDims[0] << "x" << m_iDims[1] << "x" << m_iDims[2];
	if (getType() == CData3D::PROJECTION) res << " sinogram data \t";
	if (getType() == CData3D::VOLUME) res << " volume data \t";
	return res.str();
}

template<typename T>
void CDataMemory<T>::_allocateData(size_t size)
{
	ASTRA_ASSERT(m_pfData == NULL);

	// allocate contiguous block
#ifdef _MSC_VER
	m_pfData = (T*)_aligned_malloc(size * sizeof(T), 16);
#else
	int ret = posix_memalign((void**)&m_pfData, 16, size * sizeof(T));
	ASTRA_ASSERT(ret == 0);
#endif
	ASTRA_ASSERT(((size_t)m_pfData & 15) == 0);
}

template<typename T>
void CDataMemory<T>::_freeData()
{
	// free memory for data block
#ifdef _MSC_VER
	_aligned_free(m_pfData);
#else
	free(m_pfData);
#endif

	m_pfData = nullptr;
}

CFloat32ProjectionData3D *createCFloat32ProjectionData3DMemory(const CProjectionGeometry3D &geom)
{
	size_t size = geom.getProjectionCount();
	size *= geom.getDetectorTotCount();

	CDataStorage *storage = new CDataMemory<float32>(size);
	if (!storage)
		return 0;
	return new CFloat32ProjectionData3D(geom, storage);
}

CFloat32ProjectionData3D *createCFloat32ProjectionData3DMemory(std::unique_ptr<CProjectionGeometry3D> &&geom)
{
	size_t size = geom->getProjectionCount();
	size *= geom->getDetectorTotCount();

	CDataStorage *storage = new CDataMemory<float32>(size);
	if (!storage)
		return 0;
	return new CFloat32ProjectionData3D(std::move(geom), storage);
}


CFloat32VolumeData3D *createCFloat32VolumeData3DMemory(const CVolumeGeometry3D &geom)
{
	CDataStorage *storage = new CDataMemory<float32>(geom.getGridTotCount());
	if (!storage)
		return 0;
	return new CFloat32VolumeData3D(geom, storage);
}

CFloat32VolumeData3D *createCFloat32VolumeData3DMemory(std::unique_ptr<CVolumeGeometry3D> &&geom)
{
	CDataStorage *storage = new CDataMemory<float32>(geom->getGridTotCount());
	if (!storage)
		return 0;
	return new CFloat32VolumeData3D(std::move(geom), storage);
}


}
