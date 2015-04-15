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

#include "astra/Float32ProjectionData2D.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Default constructor
CFloat32ProjectionData2D::CFloat32ProjectionData2D() :
	CFloat32Data2D() 
{
	m_bInitialized = false;
	m_pGeometry = NULL;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class, allocating (but not initializing) the data block.
CFloat32ProjectionData2D::CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry) 
{
	m_bInitialized = false;
	initialize(_pGeometry);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class with initialization of the data.
CFloat32ProjectionData2D::CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry, float32* _pfData) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _pfData);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class with scalar initialization of the data.
CFloat32ProjectionData2D::CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry, float32 _fScalar) 
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _fScalar);
}


//----------------------------------------------------------------------------------------
// Copy constructor
CFloat32ProjectionData2D::CFloat32ProjectionData2D(const CFloat32ProjectionData2D& _other) : CFloat32Data2D(_other)
{
	// Data is copied by parent constructor
	m_pGeometry = _other.m_pGeometry->clone();
	m_bInitialized = true;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32ProjectionData2D class with pre-allocated data
CFloat32ProjectionData2D::CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry, CFloat32CustomMemory* _pCustomMemory)
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _pCustomMemory);
}
 


// Assignment operator

CFloat32ProjectionData2D& CFloat32ProjectionData2D::operator=(const CFloat32ProjectionData2D& _other)
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
// Initialization
bool CFloat32ProjectionData2D::initialize(CProjectionGeometry2D* _pGeometry)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorCount(), m_pGeometry->getProjectionAngleCount());
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData2D::initialize(CProjectionGeometry2D* _pGeometry, const float32* _pfData)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorCount(), m_pGeometry->getProjectionAngleCount(), _pfData);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData2D::initialize(CProjectionGeometry2D* _pGeometry, float32 _fScalar)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorCount(), m_pGeometry->getProjectionAngleCount(), _fScalar);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32ProjectionData2D::initialize(CProjectionGeometry2D* _pGeometry, CFloat32CustomMemory* _pCustomMemory) 
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getDetectorCount(), m_pGeometry->getProjectionAngleCount(), _pCustomMemory);
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Destructor
CFloat32ProjectionData2D::~CFloat32ProjectionData2D() 
{
	if (m_bInitialized)
		delete m_pGeometry;
	m_pGeometry = 0;
}

//----------------------------------------------------------------------------------------
void CFloat32ProjectionData2D::changeGeometry(CProjectionGeometry2D* _pGeometry)
{
	if (!m_bInitialized) return;

	delete m_pGeometry;
	m_pGeometry = _pGeometry->clone();
}

} // end namespace astra
