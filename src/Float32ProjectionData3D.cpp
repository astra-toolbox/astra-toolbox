/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#include "astra/Float32ProjectionData3D.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructors
//----------------------------------------------------------------------------------------


// Default constructor
CFloat32ProjectionData3D::CFloat32ProjectionData3D() :
	CFloat32Data3D() {
}

// Destructor
CFloat32ProjectionData3D::~CFloat32ProjectionData3D() {
	delete m_pGeometry;
	m_pGeometry = 0;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator+=(const CFloat32ProjectionData3D& _data)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();
#ifdef _DEBUG
	CProjectionGeometry3D * pDataGeometry = _data.getGeometry();
	int iDataProjectionDetectorCount = pDataGeometry->getDetectorTotCount();

	ASTRA_ASSERT(iProjectionCount == pDataGeometry->getProjectionCount());
	ASTRA_ASSERT(iDetectorCount == iDataProjectionDetectorCount);
#endif

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);
		CFloat32VolumeData2D * pDataProjection = _data.fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];
			float32 fDataValue = pDataProjection->getDataConst()[iDetectorIndex];

			fThisValue += fDataValue;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
		delete pDataProjection;
	}

	return *this;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator-=(const CFloat32ProjectionData3D& _data)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();
#ifdef _DEBUG
	CProjectionGeometry3D * pDataGeometry = _data.getGeometry();
	int iDataProjectionDetectorCount = pDataGeometry->getDetectorTotCount();

	ASTRA_ASSERT(iProjectionCount == pDataGeometry->getProjectionCount());
	ASTRA_ASSERT(iDetectorCount == iDataProjectionDetectorCount);
#endif

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);
		CFloat32VolumeData2D * pDataProjection = _data.fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];
			float32 fDataValue = pDataProjection->getDataConst()[iDetectorIndex];

			fThisValue -= fDataValue;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
		delete pDataProjection;
	}

	return *this;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator*=(const CFloat32ProjectionData3D& _data)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();
#ifdef _DEBUG
	CProjectionGeometry3D * pDataGeometry = _data.getGeometry();
	int iDataProjectionDetectorCount = pDataGeometry->getDetectorTotCount();

	ASTRA_ASSERT(iProjectionCount == pDataGeometry->getProjectionCount());
	ASTRA_ASSERT(iDetectorCount == iDataProjectionDetectorCount);
#endif

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);
		CFloat32VolumeData2D * pDataProjection = _data.fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];
			float32 fDataValue = pDataProjection->getDataConst()[iDetectorIndex];

			fThisValue *= fDataValue;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
		delete pDataProjection;
	}

	return *this;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator*=(const float32& _fScalar)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue *= _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator/=(const float32& _fScalar)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue /= _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator+=(const float32& _fScalar)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue += _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

CFloat32ProjectionData3D& CFloat32ProjectionData3D::operator-=(const float32& _fScalar)
{
	CProjectionGeometry3D * pThisGeometry = getGeometry();

	int iProjectionCount = pThisGeometry->getProjectionCount();
	int iDetectorCount = pThisGeometry->getDetectorTotCount();

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		CFloat32VolumeData2D * pThisProjection = fetchProjection(iProjectionIndex);

		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
			float32 fThisValue = pThisProjection->getData()[iDetectorIndex];

			fThisValue -= _fScalar;

			pThisProjection->getData()[iDetectorIndex] = fThisValue;
		}

		returnProjection(iProjectionIndex, pThisProjection);

		delete pThisProjection;
	}

	return *this;
}

void CFloat32ProjectionData3D::changeGeometry(CProjectionGeometry3D* _pGeometry)
{
	if (!m_bInitialized) return;

	delete m_pGeometry;
	m_pGeometry = _pGeometry->clone();
}


} // end namespace astra
