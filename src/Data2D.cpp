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

#include "astra/Data2D.h"

namespace astra {

std::string CData2D::description() const
{
	std::stringstream res;
	res << m_iDims[0] << "x" << m_iDims[1];
	if (getType() == CData2D::PROJECTION) res << " sinogram data \t";
	if (getType() == CData2D::VOLUME) res << " volume data \t";
	return res.str();
}

template<typename T>
void CDataMemory<T>::_allocateData(size_t size)
{
	ASTRA_ASSERT(m_pfData == NULL);
	ASTRA_ASSERT(m_bOwnData);

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
	if (!m_bOwnData)
		return;

	// free memory for data block
#ifdef _MSC_VER
	_aligned_free(m_pfData);
#else
	free(m_pfData);
#endif

	m_pfData = nullptr;
}

CData2D& CData2D::operator+=(const CData2D &_other)
{
	ASTRA_ASSERT(isFloat32Memory());
	ASTRA_ASSERT(_other.isFloat32Memory());
	ASTRA_ASSERT(getSize() == _other.getSize());
	float32 *out = getFloat32Memory();
	const float32 *in = _other.getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		out[i] += in[i];
	return (*this);
}

CData2D& CData2D::operator*=(const CData2D &_other)
{
	ASTRA_ASSERT(isFloat32Memory());
	ASTRA_ASSERT(_other.isFloat32Memory());
	ASTRA_ASSERT(getSize() == _other.getSize());
	float32 *out = getFloat32Memory();
	const float32 *in = _other.getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		out[i] *= in[i];
	return (*this);
}

void CData2D::copyData(const CData2D &_other)
{
	ASTRA_ASSERT(isFloat32Memory());
	ASTRA_ASSERT(_other.isFloat32Memory());
	ASTRA_ASSERT(getSize() == _other.getSize());
	float32 *out = getFloat32Memory();
	const float32 *in = _other.getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		out[i] = in[i];
}


CData2D& CData2D::operator*=(float32 _value)
{
	ASTRA_ASSERT(isFloat32Memory());
	float32 *out = getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		out[i] *= _value;
	return (*this);
}

void CData2D::setData(float32 _value)
{
	ASTRA_ASSERT(isFloat32Memory());
	float32 *out = getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		out[i] = _value;
}

void CData2D::clampMin(float32 _fMin)
{
	ASTRA_ASSERT(isFloat32Memory());
	float32 *out = getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		if (out[i] < _fMin)
			out[i] = _fMin;
}

void CData2D::clampMax(float32 _fMax)
{
	ASTRA_ASSERT(isFloat32Memory());
	float32 *out = getFloat32Memory();
	for (size_t i = 0; i < m_iSize; i++)
		if (out[i] > _fMax)
			out[i] = _fMax;
}

CFloat32ProjectionData2D *createCFloat32ProjectionData2DMemory(const CProjectionGeometry2D &geom)
{
	size_t size = geom.getProjectionAngleCount();
	size *= geom.getDetectorCount();

	CDataStorage *storage = new CDataMemory<float32>(size);
	if (!storage)
		return 0;
	return new CFloat32ProjectionData2D(geom, storage);
}

CFloat32ProjectionData2D *createCFloat32ProjectionData2DMemory(std::unique_ptr<CProjectionGeometry2D> &&geom)
{
	size_t size = geom->getProjectionAngleCount();
	size *= geom->getDetectorCount();

	CDataStorage *storage = new CDataMemory<float32>(size);
	if (!storage)
		return 0;
	return new CFloat32ProjectionData2D(std::move(geom), storage);
}


CFloat32VolumeData2D *createCFloat32VolumeData2DMemory(const CVolumeGeometry2D &geom)
{
	CDataStorage *storage = new CDataMemory<float32>(geom.getGridTotCount());
	if (!storage)
		return 0;
	return new CFloat32VolumeData2D(geom, storage);
}

CFloat32VolumeData2D *createCFloat32VolumeData2DMemory(std::unique_ptr<CVolumeGeometry2D> &&geom)
{
	CDataStorage *storage = new CDataMemory<float32>(geom->getGridTotCount());
	if (!storage)
		return 0;
	return new CFloat32VolumeData2D(std::move(geom), storage);
}

template class CDataMemory<float32>;
template class CData2DObject<CProjectionGeometry2D>;
template class CData2DObject<CVolumeGeometry2D>;

}
