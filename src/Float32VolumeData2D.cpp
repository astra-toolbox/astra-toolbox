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

#include "astra/Float32VolumeData2D.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor
CFloat32VolumeData2D::CFloat32VolumeData2D() :
	CFloat32Data2D() 
{
	m_pGeometry = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class, allocating (but not initializing) the data block.
CFloat32VolumeData2D::CFloat32VolumeData2D(CVolumeGeometry2D* _pGeometry) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with initialization of the data.
CFloat32VolumeData2D::CFloat32VolumeData2D(CVolumeGeometry2D* _pGeometry, float32* _pfData) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _pfData);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with initialization of the data.
CFloat32VolumeData2D::CFloat32VolumeData2D(CVolumeGeometry2D* _pGeometry, float32 _fScalar) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _fScalar);
}

//----------------------------------------------------------------------------------------
// Copy constructor
CFloat32VolumeData2D::CFloat32VolumeData2D(const CFloat32VolumeData2D& _other) : CFloat32Data2D(_other)
{
	m_pGeometry = _other.m_pGeometry->clone();
	m_bInitialized = true;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with pre-allocated data
CFloat32VolumeData2D::CFloat32VolumeData2D(CVolumeGeometry2D* _pGeometry, CFloat32CustomMemory* _pCustomMemory)
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _pCustomMemory);
}


// Assignment operator

CFloat32VolumeData2D& CFloat32VolumeData2D::operator=(const CFloat32VolumeData2D& _other)
{
	ASTRA_ASSERT(_other.m_bInitialized);

	if (m_bInitialized)
		delete m_pGeometry;
	*((CFloat32Data2D*)this) = _other;
	m_pGeometry = _other.m_pGeometry->clone();
	m_bInitialized = true;

	return *this;
}

//----------------------------------------------------------------------------------------
// Destructor
CFloat32VolumeData2D::~CFloat32VolumeData2D() 
{
	if (m_bInitialized)
		delete m_pGeometry;
	m_pGeometry = 0;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData2D::initialize(CVolumeGeometry2D* _pGeometry)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount());
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData2D::initialize(CVolumeGeometry2D* _pGeometry, const float32* _pfData)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), _pfData);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData2D::initialize(CVolumeGeometry2D* _pGeometry, float32 _fScalar)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), _fScalar);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData2D::initialize(CVolumeGeometry2D* _pGeometry, CFloat32CustomMemory* _pCustomMemory) 
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), _pCustomMemory);
	return m_bInitialized;
}


//----------------------------------------------------------------------------------------
void CFloat32VolumeData2D::changeGeometry(CVolumeGeometry2D* _pGeometry)
{
	if (!m_bInitialized) return;

	delete m_pGeometry;
	m_pGeometry = _pGeometry->clone();
}


} // end namespace astra
