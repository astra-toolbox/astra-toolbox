/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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
$Id$
*/

#include "astra/Float32VolumeData3D.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32VolumeData3D::CFloat32VolumeData3D() :
	CFloat32Data3D() {

}

//----------------------------------------------------------------------------------------
// Destructor
CFloat32VolumeData3D::~CFloat32VolumeData3D() {

}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator+=(const CFloat32VolumeData3D& _data)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();
#ifdef _DEBUG
	CVolumeGeometry3D * pDataGeometry = _data.getGeometry();
	int iThisSlicePixelCount = pThisGeometry->getGridRowCount() * pThisGeometry->getGridColCount();
	int iDataSlicePixelCount = pDataGeometry->getGridRowCount() * pDataGeometry->getGridColCount();

	ASTRA_ASSERT(iSliceCount == pDataGeometry->getGridSliceCount());
	ASTRA_ASSERT(iThisSlicePixelCount == iDataSlicePixelCount);
#endif

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);
		CFloat32VolumeData2D * pDataProjection = _data.fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];
			float32 fDataValue = pDataProjection->getDataConst()[iDetectorIndex];

			fThisValue += fDataValue;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
		delete pDataProjection;
	}

	return *this;
}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator-=(const CFloat32VolumeData3D& _data)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();
#ifdef _DEBUG
	CVolumeGeometry3D * pDataGeometry = _data.getGeometry();
	int iThisSlicePixelCount = pThisGeometry->getGridRowCount() * pThisGeometry->getGridColCount();
	int iDataSlicePixelCount = pDataGeometry->getGridRowCount() * pDataGeometry->getGridColCount();

	ASTRA_ASSERT(iSliceCount == pDataGeometry->getGridSliceCount());
	ASTRA_ASSERT(iThisSlicePixelCount == iDataSlicePixelCount);
#endif

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);
		CFloat32VolumeData2D * pDataProjection = _data.fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];
			float32 fDataValue = pDataProjection->getDataConst()[iDetectorIndex];

			fThisValue -= fDataValue;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
		delete pDataProjection;
	}

	return *this;
}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator*=(const CFloat32VolumeData3D& _data)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();
#ifdef _DEBUG
	CVolumeGeometry3D * pDataGeometry = _data.getGeometry();
	int iThisSlicePixelCount = pThisGeometry->getGridRowCount() * pThisGeometry->getGridColCount();
	int iDataSlicePixelCount = pDataGeometry->getGridRowCount() * pDataGeometry->getGridColCount();

	ASTRA_ASSERT(iSliceCount == pDataGeometry->getGridSliceCount());
	ASTRA_ASSERT(iThisSlicePixelCount == iDataSlicePixelCount);
#endif

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);
		CFloat32VolumeData2D * pDataProjection = _data.fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];
			float32 fDataValue = pDataProjection->getDataConst()[iDetectorIndex];

			fThisValue *= fDataValue;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
		delete pDataProjection;
	}

	return *this;
}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator*=(const float32& _fScalar)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue *= _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator/=(const float32& _fScalar)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue /= _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator+=(const float32& _fScalar)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue += _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

CFloat32VolumeData3D& CFloat32VolumeData3D::operator-=(const float32& _fScalar)
{
	CVolumeGeometry3D * pThisGeometry = getGeometry();

	int iSliceCount = pThisGeometry->getGridSliceCount();

	for(int iSliceIndex = 0; iSliceIndex < iSliceCount; iSliceIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchSliceZ(iSliceIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorIndex; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue -= _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnSliceZ(iSliceIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

} // end namespace astra
