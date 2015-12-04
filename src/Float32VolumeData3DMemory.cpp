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

#include "astra/Float32VolumeData3DMemory.h"

#include <cstring>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor
CFloat32VolumeData3DMemory::CFloat32VolumeData3DMemory() :
	CFloat32Data3DMemory() 
{
	m_pGeometry       = NULL;
	m_bInitialized    = false;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class, allocating (but not initializing) the data block.
CFloat32VolumeData3DMemory::CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry) 
{
	m_bInitialized    = false;
	m_bInitialized    = initialize(_pGeometry);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with initialization of the data.
CFloat32VolumeData3DMemory::CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry, const float32* _pfData) 
{
	m_bInitialized    = false;
	m_bInitialized    = initialize(_pGeometry, _pfData);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with initialization of the data.
CFloat32VolumeData3DMemory::CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry, float32 _fScalar) 
{
	m_bInitialized    = false;
	m_bInitialized    = initialize(_pGeometry, _fScalar);
}
//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with pre-allocated data
CFloat32VolumeData3DMemory::CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory)
{
	m_bInitialized    = false;
	m_bInitialized    = initialize(_pGeometry, _pCustomMemory);
}


//----------------------------------------------------------------------------------------
// Destructor
CFloat32VolumeData3DMemory::~CFloat32VolumeData3DMemory() 
{
	if(m_pGeometry){
		delete m_pGeometry;
	}
	m_pGeometry = 0;

}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData3DMemory::initialize(CVolumeGeometry3D* _pGeometry)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), m_pGeometry->getGridSliceCount());
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData3DMemory::initialize(CVolumeGeometry3D* _pGeometry, const float32* _pfData)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), m_pGeometry->getGridSliceCount(), _pfData);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData3DMemory::initialize(CVolumeGeometry3D* _pGeometry, float32 _fScalar)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), m_pGeometry->getGridSliceCount(), _fScalar);
	return m_bInitialized;
}
//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData3DMemory::initialize(CVolumeGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory) 
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), m_pGeometry->getGridSliceCount(), _pCustomMemory);
	return m_bInitialized;
}


//----------------------------------------------------------------------------------------
// Fetch a slice
CFloat32VolumeData2D * CFloat32VolumeData3DMemory::fetchSliceZ(int _iSliceIndex) const
{
	// fetch slice of the geometry
	int iRowCount = m_pGeometry->getGridRowCount();
	int iColumnCount = m_pGeometry->getGridColCount();
	CVolumeGeometry2D volGeom(iColumnCount, iRowCount);

	// create new volume data
	CFloat32VolumeData2D* res = new CFloat32VolumeData2D(&volGeom);

	// copy data
	int iSliceCount = m_pGeometry->getGridSliceCount();
	float * pfTargetData = res->getData();
	for(int iRowIndex = 0; iRowIndex < iRowCount; iRowIndex++)
	{
		for(int iColumnIndex = 0; iColumnIndex < iColumnCount; iColumnIndex++)
		{
			int iSourceIndex = _iSliceIndex * iColumnCount * iRowCount + iRowIndex * iColumnCount + iColumnIndex;
			int iTargetIndex = iRowIndex * iColumnCount + iColumnIndex;
			float fStoredValue = m_pfData[iSourceIndex];
			pfTargetData[iTargetIndex] = fStoredValue;
		}
	}
	// return
	return res;
}

//----------------------------------------------------------------------------------------
// Return a slice
void CFloat32VolumeData3DMemory::returnSliceZ(int _iSliceIndex, CFloat32VolumeData2D * _pSlice)
{
	int iRowCount = _pSlice->getGeometry()->getGridRowCount();
	int iColumnCount = _pSlice->getGeometry()->getGridColCount();

	assert(iRowCount == m_pGeometry->getGridRowCount());
	assert(iColumnCount == m_pGeometry->getGridColCount());

	for(int iRowIndex = 0; iRowIndex < iRowCount; iRowIndex++)
	{
		for(int iColumnIndex = 0; iColumnIndex < iColumnCount; iColumnIndex++)
		{
			int iSourceIndex = iRowIndex * iColumnCount + iColumnIndex;
			int iTargetIndex = _iSliceIndex * iColumnCount * iRowCount + iRowIndex * iColumnCount + iColumnIndex;
			float fStoredValue = _pSlice->getDataConst()[iSourceIndex];
			m_pfData[iTargetIndex] = fStoredValue;
		}
	}
}

CFloat32VolumeData2D * CFloat32VolumeData3DMemory::fetchSliceX(int _iColumnIndex) const
{
	// TODO:
	assert(false);
	return NULL;
}

CFloat32VolumeData2D * CFloat32VolumeData3DMemory::fetchSliceY(int _iRowIndex) const
{
	// TODO:
	assert(false);
	return NULL;
}

void CFloat32VolumeData3DMemory::returnSliceX(int _iColumnIndex, CFloat32VolumeData2D * _pSliceData)
{
	// TODO:
	assert(false);
}

void CFloat32VolumeData3DMemory::returnSliceY(int _iRowIndex, CFloat32VolumeData2D * _pSliceData)
{
	// TODO:
	assert(false);
}

//----------------------------------------------------------------------------------------
// Returns a specific value
float32 CFloat32VolumeData3DMemory::getVoxelValue(int _iIndex)
{
	return m_pfData[_iIndex];
}

//----------------------------------------------------------------------------------------
// Sets a specific value
void CFloat32VolumeData3DMemory::setVoxelValue(int _iIndex, float32 _fValue)
{
	m_pfData[_iIndex] = _fValue;
}
//----------------------------------------------------------------------------------------

CFloat32VolumeData3DMemory& CFloat32VolumeData3DMemory::operator=(const CFloat32VolumeData3DMemory& _dataIn)
{
	memcpy(m_pfData, _dataIn.m_pfData, sizeof(float32) * _dataIn.m_pGeometry->getGridTotCount());

	return *this;
}

} // end namespace astra
