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

#include "astra/Float32Data2D.h"
#include <iostream>
#include <cstring>
#include <sstream>

#ifdef _MSC_VER
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace astra {

CFloat32CustomMemory::~CFloat32CustomMemory() {

}


  //----------------------------------------------------------------------------------------
 // Constructors
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32Data2D::CFloat32Data2D()
{
	_clear();
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32Data2D class, allocating (but not initializing) the data block.
CFloat32Data2D::CFloat32Data2D(int _iWidth, int _iHeight) 
{
	m_bInitialized = false;
	_initialize(_iWidth, _iHeight);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32Data2D class with initialization of the data block. 
CFloat32Data2D::CFloat32Data2D(int _iWidth, int _iHeight, const float32* _pfData)
{
	m_bInitialized = false;
	_initialize(_iWidth, _iHeight, _pfData);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32Data2D class with initialization of the data block. 
CFloat32Data2D::CFloat32Data2D(int _iWidth, int _iHeight, float32 _fScalar)
{
	m_bInitialized = false;
	_initialize(_iWidth, _iHeight, _fScalar);
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32Data2D class with pre-allocated memory. 
CFloat32Data2D::CFloat32Data2D(int _iWidth, int _iHeight, CFloat32CustomMemory *_pCustomMemory)
{
	m_bInitialized = false;
	_initialize(_iWidth, _iHeight, _pCustomMemory);
}

//----------------------------------------------------------------------------------------
// Copy constructor
CFloat32Data2D::CFloat32Data2D(const CFloat32Data2D& _other)
{
	m_bInitialized = false;
	*this = _other;
}

//----------------------------------------------------------------------------------------
// Assignment operator
CFloat32Data2D& CFloat32Data2D::operator=(const CFloat32Data2D& _dataIn)
{
	ASTRA_ASSERT(_dataIn.m_bInitialized);

	if (m_bInitialized) {
		if (m_iWidth == _dataIn.m_iWidth && m_iHeight == _dataIn.m_iHeight) {
			// Same dimensions, so no need to re-allocate memory

			m_fGlobalMin = _dataIn.m_fGlobalMin;
			m_fGlobalMax = _dataIn.m_fGlobalMax;
			m_fGlobalMean = _dataIn.m_fGlobalMean;

			ASTRA_ASSERT(m_iSize == (size_t)m_iWidth * m_iHeight);
			ASTRA_ASSERT(m_pfData);

			memcpy(m_pfData, _dataIn.m_pfData, m_iSize * sizeof(float32));
		} else {
			if (m_pCustomMemory) {
				// Can't re-allocate custom data
				ASTRA_ASSERT(false);
				return *(CFloat32Data2D*)0;
			}
			// Re-allocate data
			_unInit();
			_initialize(_dataIn.getWidth(), _dataIn.getHeight(), _dataIn.getDataConst());
		}
	} else {
		_initialize(_dataIn.getWidth(), _dataIn.getHeight(), _dataIn.getDataConst());
	}

	return (*this);
}

//----------------------------------------------------------------------------------------
// Destructor. Free allocated memory
CFloat32Data2D::~CFloat32Data2D() 
{
	if (m_bInitialized)
	{
		_unInit();
	}
}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class, allocating (but not initializing) the data block.
bool CFloat32Data2D::_initialize(int _iWidth, int _iHeight)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);

	if (m_bInitialized)
	{
		_unInit();
	}
	
	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iSize = (size_t)m_iWidth * m_iHeight;

	// allocate memory for the data, but do not fill it
	m_pfData = 0;
	m_ppfData2D = 0;
	m_pCustomMemory = 0;
	_allocateData();

	// set minmax to default values
	m_fGlobalMin = 0.0;
	m_fGlobalMax = 0.0;
	m_fGlobalMean = 0.0;

	// initialization complete
	return true;

}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class with initialization of the data block. 
bool CFloat32Data2D::_initialize(int _iWidth, int _iHeight, const float32 *_pfData)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_pfData != NULL);

	if (m_bInitialized)
	{
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iSize = (size_t)m_iWidth * m_iHeight;

	// allocate memory for the data 
	m_pfData = 0;
	m_ppfData2D = 0;
	m_pCustomMemory = 0;
	_allocateData();

	// fill the data block with a copy of the input data
	size_t i;
	for (i = 0; i < m_iSize; ++i) {
		m_pfData[i] = _pfData[i];
	}

	// initialization complete
	return true;
}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class with a scalar initialization of the data block. 
bool CFloat32Data2D::_initialize(int _iWidth, int _iHeight, float32 _fScalar)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);

	if (m_bInitialized)	{
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iSize = (size_t)m_iWidth * m_iHeight;

	// allocate memory for the data 
	m_pfData = 0;
	m_ppfData2D = 0;
	m_pCustomMemory = 0;
	_allocateData();

	// fill the data block with a copy of the input data
	size_t i;
	for (i = 0; i < m_iSize; ++i)
	{
		m_pfData[i] = _fScalar;
	}

	// initialization complete
	return true;
}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class with pre-allocated memory
bool CFloat32Data2D::_initialize(int _iWidth, int _iHeight, CFloat32CustomMemory* _pCustomMemory)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_pCustomMemory != NULL);

	if (m_bInitialized)
	{
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iSize = (size_t)m_iWidth * m_iHeight;

	// initialize the data pointers
	m_pCustomMemory = _pCustomMemory;
	m_pfData = 0;
	m_ppfData2D = 0;
	_allocateData();

	// initialization complete
	return true;
}



  //----------------------------------------------------------------------------------------
 // Memory Allocation 
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Allocate memory for m_pfData and m_ppfData2D arrays.
void CFloat32Data2D::_allocateData()
{
	// basic checks
	ASTRA_ASSERT(!m_bInitialized);

	ASTRA_ASSERT(m_iSize > 0);
	ASTRA_ASSERT(m_iSize == (size_t)m_iWidth * m_iHeight);
	ASTRA_ASSERT(m_pfData == NULL);
	ASTRA_ASSERT(m_ppfData2D == NULL);

	if (!m_pCustomMemory) {

		// allocate contiguous block
#ifdef _MSC_VER
		m_pfData = (float32*)_aligned_malloc(m_iSize * sizeof(float32), 16);
#else
		int ret = posix_memalign((void**)&m_pfData, 16, m_iSize * sizeof(float32));
		ASTRA_ASSERT(ret == 0);
#endif

	} else {
		m_pfData = m_pCustomMemory->m_fPtr;
	}

	// create array of pointers to each row of the data block
	m_ppfData2D = new float32*[m_iHeight];
	for (int iy = 0; iy < m_iHeight; iy++)
	{
		m_ppfData2D[iy] = &(m_pfData[iy * m_iWidth]);
	}
}

//----------------------------------------------------------------------------------------
// Free memory for m_pfData and m_ppfData2D arrays.
void CFloat32Data2D::_freeData()
{
	// basic checks
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_ppfData2D != NULL);
	// free memory for index table
	delete[] m_ppfData2D;

	if (!m_pCustomMemory) {
		// free memory for data block
#ifdef _MSC_VER
		_aligned_free(m_pfData);
#else
		free(m_pfData);
#endif
	} else {
		delete m_pCustomMemory;
		m_pCustomMemory = 0;
	}
}


//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
void CFloat32Data2D::_clear()
{
	m_iWidth = 0;
	m_iHeight = 0;
	m_iSize = 0;

	m_pfData = NULL;
	m_ppfData2D = NULL;
	m_pCustomMemory = NULL;

	m_fGlobalMin = 0.0f;
	m_fGlobalMax = 0.0f;
}

//----------------------------------------------------------------------------------------
// Un-initialize the object, bringing it back in the unitialized state.
void CFloat32Data2D::_unInit()
{
	ASTRA_ASSERT(m_bInitialized);

	_freeData();
	_clear();
	m_bInitialized = false;
}
//----------------------------------------------------------------------------------------



  //----------------------------------------------------------------------------------------
 // Data Operations
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Copy the data block pointed to by _pfData to the data block pointed to by m_pfData.
void CFloat32Data2D::copyData(const float32* _pfData)
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_pfData != NULL);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);

	// copy data
	size_t i;
	for (i = 0; i < m_iSize; ++i) {
		m_pfData[i] = _pfData[i];
	}
}	

//----------------------------------------------------------------------------------------
// scale m_pfData from 0 to 255.

void CFloat32Data2D::scale() 
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);

	_computeGlobalMinMax();
	for (size_t i = 0; i < m_iSize; i++) 
	{
		// do checks
		m_pfData[i]= (m_pfData[i] - m_fGlobalMin) / (m_fGlobalMax - m_fGlobalMin) * 255; ;
	}


}

//----------------------------------------------------------------------------------------
// Set each element of the data to a specific scalar value
void CFloat32Data2D::setData(float32 _fScalar)
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);

	// copy data
	size_t i;
	for (i = 0; i < m_iSize; ++i)
	{
		m_pfData[i] = _fScalar;
	}
}

//----------------------------------------------------------------------------------------
// Clear Data
void CFloat32Data2D::clearData() 
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);
	
	// set data
	size_t i;
	for (i = 0; i < m_iSize; ++i) {
		m_pfData[i] = 0.0f;
	}
}
//----------------------------------------------------------------------------------------



  //----------------------------------------------------------------------------------------
 // Statistics Operations
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Update data statistics, such as minimum and maximum value, after the data has been modified. 
void CFloat32Data2D::updateStatistics()
{
	_computeGlobalMinMax();
}

//----------------------------------------------------------------------------------------
// Find the minimum and maximum data value.
void CFloat32Data2D::_computeGlobalMinMax() 
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);
	
	// initial values
	m_fGlobalMin = m_pfData[0];
	m_fGlobalMax = m_pfData[0];
	m_fGlobalMean = 0.0f;

	// loop
	for (size_t i = 0; i < m_iSize; i++) 
	{
		// do checks
		float32 v = m_pfData[i];
		if (v < m_fGlobalMin) {
			m_fGlobalMin = v;
		}
		if (v > m_fGlobalMax) {
			m_fGlobalMax = v;
		}
		m_fGlobalMean +=v;
	}
	m_fGlobalMean /= m_iSize;
}
//----------------------------------------------------------------------------------------


CFloat32Data2D& CFloat32Data2D::clampMin(float32& _fMin)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		if (m_pfData[i] < _fMin)
			m_pfData[i] = _fMin;
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::clampMax(float32& _fMax)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		if (m_pfData[i] > _fMax)
			m_pfData[i] = _fMax;
	}
	return (*this);
}


//----------------------------------------------------------------------------------------
// Operator Overloading 
//----------------------------------------------------------------------------------------
CFloat32Data2D& CFloat32Data2D::operator+=(const CFloat32Data2D& v)
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(v.m_bInitialized);
	ASTRA_ASSERT(getSize() == v.getSize());
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] += v.m_pfData[i]; 
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::operator-=(const CFloat32Data2D& v)
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(v.m_bInitialized);
	ASTRA_ASSERT(getSize() == v.getSize());
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] -= v.m_pfData[i]; 
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::operator*=(const CFloat32Data2D& v)
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(v.m_bInitialized);
	ASTRA_ASSERT(getSize() == v.getSize());
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] *= v.m_pfData[i]; 
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::operator*=(const float32& f)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] *= f; 
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::operator/=(const float32& f)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] /= f; 
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::operator+=(const float32& f)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] += f;
	}
	return (*this);
}

CFloat32Data2D& CFloat32Data2D::operator-=(const float32& f)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		m_pfData[i] -= f;
	}
	return (*this);
}


std::string CFloat32Data2D::description() const
{
	std::stringstream res;
	res << m_iWidth << "x" << m_iHeight;
	if (getType() == CFloat32Data2D::PROJECTION) res << " sinogram data \t";
	if (getType() == CFloat32Data2D::VOLUME) res << " volume data \t";
	return res.str();
}




} // end namespace astra
