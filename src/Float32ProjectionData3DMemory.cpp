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

#include "astra/Float32ProjectionData3DMemory.h"
#include "astra/ParallelProjectionGeometry3D.h"

#include <cstring>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor
CFloat32ProjectionData3DMemory::CFloat32ProjectionData3DMemory() :
	CFloat32Data3DMemory() 
{
	m_bInitialized = false;
	m_pGeometry = NULL;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class, allocating (but not initializing) the data block.
CFloat32ProjectionData3DMemory::CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry) 
{
	m_bInitialized = false;
	initialize(_pGeometry);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class with initialization of the data.
CFloat32ProjectionData3DMemory::CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry, float32* _pfData) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _pfData);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class with initialization of the data.
CFloat32ProjectionData3DMemory::CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry, float32 _fScalar) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _fScalar);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class with pre-allocated data
CFloat32ProjectionData3DMemory::CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory)
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _pCustomMemory);
}
 

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData3DMemory::initialize(CProjectionGeometry3D* _pGeometry)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorColCount(),		// width
								 m_pGeometry->getProjectionCount(),	// height
								 m_pGeometry->getDetectorRowCount());		// depth
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData3DMemory::initialize(CProjectionGeometry3D* _pGeometry, const float32* _pfData)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorColCount(),		// width
								 m_pGeometry->getProjectionCount(),	// height
								 m_pGeometry->getDetectorRowCount(),		// depth
								 _pfData);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData3DMemory::initialize(CProjectionGeometry3D* _pGeometry, float32 _fScalar)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorColCount(),		// width
								 m_pGeometry->getProjectionCount(),	// height
								 m_pGeometry->getDetectorRowCount(),		// depth
								 _fScalar);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData3DMemory::initialize(CProjectionGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory) 
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorColCount(), m_pGeometry->getProjectionCount(), m_pGeometry->getDetectorRowCount(), _pCustomMemory);
	return m_bInitialized;
}


//----------------------------------------------------------------------------------------
// Destructor
CFloat32ProjectionData3DMemory::~CFloat32ProjectionData3DMemory() 
{
	//delete m_pGeometry; //delete geom inherited from CFloat32ProjectionData3D
	//_unInit(); //delete stuff inherited from CFloat32Data3DMemory
}

//----------------------------------------------------------------------------------------
// Fetch a projection
CFloat32VolumeData2D* CFloat32ProjectionData3DMemory::fetchProjection(int _iProjectionNr) const
{
	// fetch slice of the geometry
	CVolumeGeometry2D volGeom(m_pGeometry->getDetectorColCount(), m_pGeometry->getDetectorRowCount());
	// create new volume data
	CFloat32VolumeData2D* res = new CFloat32VolumeData2D(&volGeom);
	// copy data
	int row, col;
	for (row = 0; row < m_pGeometry->getDetectorRowCount(); ++row) {
		for (col = 0; col < m_pGeometry->getDetectorColCount(); ++col) {
			res->getData()[row*m_pGeometry->getDetectorColCount() + col] = 
				m_pfData[_iProjectionNr * m_pGeometry->getDetectorColCount() + m_pGeometry->getDetectorColCount()* m_pGeometry->getProjectionCount() * row + col];
		}
	}
	// return
	return res;
}

//----------------------------------------------------------------------------------------
// Return a projection
void CFloat32ProjectionData3DMemory::returnProjection(int _iProjectionNr, CFloat32VolumeData2D* _pProjection) 
{
	/// TODO: check geometry
	// copy data
	int row, col;
	for (row = 0; row < m_pGeometry->getDetectorRowCount(); ++row) {
		for (col = 0; col < m_pGeometry->getDetectorColCount(); ++col) {
			m_pfData[_iProjectionNr * m_pGeometry->getDetectorColCount() + m_pGeometry->getDetectorColCount()* m_pGeometry->getProjectionCount() * row + col] = 
				_pProjection->getData()[row*m_pGeometry->getDetectorColCount() + col];
		}
	}
}

//----------------------------------------------------------------------------------------
// Fetch a sinogram
CFloat32ProjectionData2D* CFloat32ProjectionData3DMemory::fetchSinogram(int _iSliceNr) const
{
	CParallelProjectionGeometry3D * pParallelProjGeo = (CParallelProjectionGeometry3D *)m_pGeometry;
	CParallelProjectionGeometry2D * pProjGeo2D = pParallelProjGeo->createProjectionGeometry2D();

	// create new projection data
	CFloat32ProjectionData2D* res = new CFloat32ProjectionData2D(pProjGeo2D);
	// copy data
	int row, col;

	int iDetectorColumnCount = m_pGeometry->getDetectorColCount();
	int iProjectionAngleCount = m_pGeometry->getProjectionCount();

	for (row = 0; row < m_pGeometry->getProjectionCount(); ++row) {
		for (col = 0; col < m_pGeometry->getDetectorColCount(); ++col)
		{
			int iTargetIndex = row * iDetectorColumnCount + col;
			int iSourceIndex = _iSliceNr * iDetectorColumnCount * iProjectionAngleCount + row * iDetectorColumnCount + col;

			float32 fStoredValue = m_pfData[iSourceIndex];

			res->getData()[iTargetIndex] = fStoredValue;
		}
	}

	delete pProjGeo2D;

	// return
	return res;
}

//----------------------------------------------------------------------------------------
// Return a sinogram
void CFloat32ProjectionData3DMemory::returnSinogram(int _iSliceNr, CFloat32ProjectionData2D* _pSinogram2D) 
{
	/// TODO: check geometry
	// copy data
	int row, col;
	for (row = 0; row < m_pGeometry->getProjectionCount(); ++row) {
		for (col = 0; col < m_pGeometry->getDetectorColCount(); ++col) {
			m_pfData[_iSliceNr*m_pGeometry->getDetectorColCount()*m_pGeometry->getProjectionCount() + row*m_pGeometry->getDetectorColCount() + col] =
				_pSinogram2D->getData()[row*m_pGeometry->getDetectorColCount() + col];
		}
	}
}

//----------------------------------------------------------------------------------------
// Returns a specific value
float32 CFloat32ProjectionData3DMemory::getDetectorValue(int _iIndex)
{
	return m_pfData[_iIndex];
}

//----------------------------------------------------------------------------------------
// Sets a specific value
void CFloat32ProjectionData3DMemory::setDetectorValue(int _iIndex, float32 _fValue)
{
	m_pfData[_iIndex] = _fValue;
}
//----------------------------------------------------------------------------------------

CFloat32ProjectionData3DMemory& CFloat32ProjectionData3DMemory::operator=(const CFloat32ProjectionData3DMemory& _dataIn)
{
	memcpy(m_pfData, _dataIn.m_pfData, sizeof(float32) * _dataIn.m_pGeometry->getDetectorTotCount());

	return *this;
}

} // end namespace astra
