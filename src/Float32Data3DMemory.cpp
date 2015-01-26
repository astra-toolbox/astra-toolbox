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

#include "astra/Float32Data3DMemory.h"
#include <iostream>

namespace astra {

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32Data3DMemory::CFloat32Data3DMemory()
{
	_clear();
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor. Free allocated memory
CFloat32Data3DMemory::~CFloat32Data3DMemory() 
{
	if (m_bInitialized)
	{
		_unInit();
	}
}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class, allocating (but not initializing) the data block.
bool CFloat32Data3DMemory::_initialize(int _iWidth, int _iHeight, int _iDepth)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_iDepth > 0);

	if (m_bInitialized)
	{
		_unInit();
	}
	
	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iDepth = _iDepth;
	m_iSize = (size_t)m_iWidth * m_iHeight * m_iDepth;

	// allocate memory for the data, but do not fill it
	m_pfData = NULL;
	m_ppfDataRowInd = NULL;
	m_pppfDataSliceInd = NULL;
	m_pCustomMemory = 0;
	_allocateData();

	// set minmax to default values
	m_fGlobalMin = 0.0;
	m_fGlobalMax = 0.0;

	// initialization complete
	return true;

}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class with initialization of the data block. 
bool CFloat32Data3DMemory::_initialize(int _iWidth, int _iHeight, int _iDepth, const float32* _pfData)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_iDepth > 0);
	ASTRA_ASSERT(_pfData != NULL);

	if (m_bInitialized) {
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iDepth = _iDepth;
	m_iSize = (size_t)m_iWidth * m_iHeight * m_iDepth;

	// allocate memory for the data, but do not fill it
	m_pfData = NULL;
	m_ppfDataRowInd = NULL;
	m_pppfDataSliceInd = NULL;
	m_pCustomMemory = 0;
	_allocateData();

	// fill the data block with a copy of the input data
	size_t i;
	for (i = 0; i < m_iSize; ++i)
	{
		m_pfData[i] = _pfData[i];
	}

	// initialization complete
	return true;
}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data2D class with initialization of the data block. 
bool CFloat32Data3DMemory::_initialize(int _iWidth, int _iHeight, int _iDepth, float32 _fScalar)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_iDepth > 0);

	if (m_bInitialized) {
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iDepth = _iDepth;
	m_iSize = (size_t)m_iWidth * m_iHeight * m_iDepth;

	// allocate memory for the data, but do not fill it
	m_pfData = NULL;
	m_ppfDataRowInd = NULL;
	m_pppfDataSliceInd = NULL;
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
// Initializes an instance of the CFloat32Data3DMemory class with pre-allocated memory
bool CFloat32Data3DMemory::_initialize(int _iWidth, int _iHeight, int _iDepth, CFloat32CustomMemory* _pCustomMemory)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_iDepth > 0);
	ASTRA_ASSERT(_pCustomMemory != NULL);

	if (m_bInitialized) {
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iDepth = _iDepth;
	m_iSize = (size_t)m_iWidth * m_iHeight * m_iDepth;

	// allocate memory for the data, but do not fill it
	m_pCustomMemory = _pCustomMemory;
	m_pfData = NULL;
	m_ppfDataRowInd = NULL;
	m_pppfDataSliceInd = NULL;
	_allocateData();

	// initialization complete
	return true;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Allocate memory for m_pfData and m_ppfData2D arrays.
void CFloat32Data3DMemory::_allocateData()
{
	// basic checks
	ASTRA_ASSERT(!m_bInitialized);

	ASTRA_ASSERT(m_iSize > 0);
	ASTRA_ASSERT(m_iSize == (size_t)m_iWidth * m_iHeight * m_iDepth);
	ASTRA_ASSERT(m_pfData == NULL);
	ASTRA_ASSERT(m_ppfDataRowInd == NULL);
	ASTRA_ASSERT(m_pppfDataSliceInd == NULL);

	if (!m_pCustomMemory) {
		// allocate contiguous block
#ifdef _MSC_VER
		m_pfData = (float32*)_aligned_malloc(m_iSize * sizeof(float32), 16);
#else
		int ret = posix_memalign((void**)&m_pfData, 16, m_iSize * sizeof(float32));
		ASTRA_ASSERT(ret == 0);
#endif
		ASTRA_ASSERT(((size_t)m_pfData & 15) == 0);
	} else {
		m_pfData = m_pCustomMemory->m_fPtr;
	}

	// create array of pointers to each row of the data block
	m_ppfDataRowInd = new float32*[m_iHeight*m_iDepth];
	for (int iy = 0; iy < m_iHeight*m_iDepth; iy++)
	{
		m_ppfDataRowInd[iy] = &(m_pfData[iy * m_iWidth]);
	}

	// create array of pointers to each row of the data block
	m_pppfDataSliceInd = new float32**[m_iDepth];
	for (int iy = 0; iy < m_iDepth; iy++)
	{
		m_pppfDataSliceInd[iy] = &(m_ppfDataRowInd[iy * m_iHeight]);
	}
}

//----------------------------------------------------------------------------------------
// Free memory for m_pfData and m_ppfData2D arrays.
void CFloat32Data3DMemory::_freeData()
{
	// basic checks
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_ppfDataRowInd != NULL);
	ASTRA_ASSERT(m_pppfDataSliceInd != NULL);

	// free memory for index table
	delete[] m_pppfDataSliceInd;
	// free memory for index table
	delete[] m_ppfDataRowInd;

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
void CFloat32Data3DMemory::_clear()
{
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = 0;

	m_pfData = NULL;
	m_ppfDataRowInd = NULL;
	m_pppfDataSliceInd = NULL;
	m_pCustomMemory = NULL;

	//m_fGlobalMin = 0.0f;
	//m_fGlobalMax = 0.0f;
}

//----------------------------------------------------------------------------------------
// Un-initialize the object, bringing it back in the unitialized state.
void CFloat32Data3DMemory::_unInit()
{
	ASTRA_ASSERT(m_bInitialized);

	_freeData();
	_clear();
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Update data statistics, such as minimum and maximum value, after the data has been modified. 
void CFloat32Data3DMemory::updateStatistics()
{
	_computeGlobalMinMax();
}

//----------------------------------------------------------------------------------------
// Find the minimum and maximum data value.
void CFloat32Data3DMemory::_computeGlobalMinMax() 
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);
	
	// initial values
	m_fGlobalMin = m_pfData[0];
	m_fGlobalMax = m_pfData[0];

	// loop
	size_t i;
	float32 v;
	for (i = 0; i < m_iSize; ++i) 
	{
		v = m_pfData[i];
		if (v < m_fGlobalMin) m_fGlobalMin = v;
		if (v > m_fGlobalMax) m_fGlobalMax = v;
	}
}

//----------------------------------------------------------------------------------------
// Copy the data block pointed to by _pfData to the data block pointed to by m_pfData.
void CFloat32Data3DMemory::copyData(const float32* _pfData, size_t _iSize)
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_pfData != NULL);
	ASTRA_ASSERT(m_pfData != NULL);
	ASTRA_ASSERT(m_iSize > 0);
	ASTRA_ASSERT(m_iSize == _iSize);

	// copy data
	size_t i;
	for (i = 0; i < m_iSize; ++i)
	{
		m_pfData[i] = _pfData[i];
	}
}

//----------------------------------------------------------------------------------------
// Copy the data block pointed to by _pfData to the data block pointed to by m_pfData.
void CFloat32Data3DMemory::setData(float32 _fScalar)
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
void CFloat32Data3DMemory::clearData() 
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

CFloat32Data3D& CFloat32Data3DMemory::clampMin(float32& _fMin)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		if (m_pfData[i] < _fMin)
			m_pfData[i] = _fMin;
	}
	return (*this);
}

CFloat32Data3D& CFloat32Data3DMemory::clampMax(float32& _fMax)
{
	ASTRA_ASSERT(m_bInitialized);
	for (size_t i = 0; i < m_iSize; i++) {
		if (m_pfData[i] > _fMax)
			m_pfData[i] = _fMax;
	}
	return (*this);
}




} // end namespace astra
