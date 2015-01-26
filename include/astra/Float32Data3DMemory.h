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

#ifndef _INC_ASTRA_FLOAT32DATA3DMEMORY
#define _INC_ASTRA_FLOAT32DATA3DMEMORY

#include "Globals.h"
#include "Float32Data3D.h"

namespace astra {

/** 
 * This class represents a three-dimensional block of float32ing point data.
 * It contains member functions for accessing this data and for performing 
 * elementary computations on the data.
 * The data block is "owned" by the class, meaning that the class is 
 * responsible for deallocation of the memory involved. 
 */
class _AstraExport CFloat32Data3DMemory : public virtual CFloat32Data3D {

protected:

	/** Pointer to the data block, represented as a 1-dimensional array.
	 * Note that the data memory is "owned" by this class, meaning that the 
	 * class is responsible for deallocation of the memory involved.
	 * To access element (ix, iy, iz) internally, use 
	 * m_pData[iz * m_iWidth * m_iHeight + iy * m_iWidth + ix] 
	 */
	 float32* m_pfData;	

	/** Array of float32 pointers, each pointing to a single row 
	 * in the m_pfData memory block.
	 * To access element (ix, iy, iz) internally, use m_ppfDataRowInd[iz * m_iHeight + iy][ix]
	*/
	float32** m_ppfDataRowInd;

	/** Array of float32 pointers, each pointing to a single slice 
	 * in the m_pfData memory block.
	 * To access element (ix, iy, iz) internally, use m_pppfDataSliceInd[iz][iy][ix]
	*/
	float32*** m_pppfDataSliceInd;	

	float32 m_fGlobalMin;	///< minimum value of the data
	float32 m_fGlobalMax;	///< maximum value of the data

	/** Allocate memory for m_pfData, m_ppfDataRowInd and m_pppfDataSliceInd arrays.
	 *
	 * The allocated block consists of m_iSize float32s. The block is
	 * not cleared after allocation and its contents is undefined. 
	 * This function may NOT be called if memory has already been allocated.
	 */
	void _allocateData();

	/** Free memory for m_pfData, m_ppfDataRowInd and m_pppfDataSliceInd arrays.
	 *
	 * This function may ONLY be called if the memory for both blocks has been 
	 * allocated before.
	*/
	void _freeData();

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 */
	void _clear();

	/** Un-initialize the object, bringing it back in the unitialized state.
	 */
	void _unInit();

	/** Find the minimum and maximum data value and store them in 
	 * m_fGlobalMin and m_fGlobalMax
	 */
	void _computeGlobalMinMax();

	/** Initialization. Initializes an instance of the CFloat32Data3DMemory class, without filling the data block.
	 * Can only be called by derived classes.
	 *
	 * Initializes an instance of the CFloat32Data3DMemory class. Memory is allocated for the 
	 * data block. The allocated memory is not cleared and its contents after 
	 * construction is undefined. Initialization may be followed by a call to 
	 * copyData() to fill the memory block. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 * This function does not set m_bInitialized to true if everything is ok.
	 *
	 * @param _iWidth width of the 3D data (x-axis), must be > 0
	 * @param _iHeight height of the 3D data (y-axis), must be > 0 
	 * @param _iDepth depth of the 3D data (z-axis), must be > 0 
	 * @return initialization of the base class successfull
	 */
	bool _initialize(int _iWidth, int _iHeight, int _iDepth);

	/** Initialization. Initializes an instance of the CFloat32Data3DMemory class with initialization of the data block.
	 * Can only be called by derived classes.
	 *
	 * Initializes an instance of the CFloat32Data3DMemory class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 * This function does not set m_bInitialized to true if everything is ok.
	 *
	 * @param _iWidth width of the 2D data (x-axis), must be > 0
	 * @param _iHeight height of the 2D data (y-axis), must be > 0 
	 * @param _iDepth depth of the 2D data (z-axis), must be > 0 
	 * @param _pfData pointer to a one-dimensional float32 data block
	 * @return initialization of the base class successfull
	 */
	bool _initialize(int _iWidth, int _iHeight, int _iDepth, const float32* _pfData);

	/** Initialization. Initializes an instance of the CFloat32Data3DMemory class with initialization of the data block.
	 * Can only be called by derived classes.
	 *
	 * Initializes an instance of the CFloat32Data3DMemory class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 * This function does not set m_bInitialized to true if everything is ok.
	 *
	 * @param _iWidth width of the 2D data (x-axis), must be > 0
	 * @param _iHeight height of the 2D data (y-axis), must be > 0 
	 * @param _iDepth depth of the 2D data (z-axis), must be > 0 
	 * @param _fScalar scalar value to fill the data
	 * @return initialization of the base class successfull
	 */
	bool _initialize(int _iWidth, int _iHeight, int _iDepth, float32 _fScalar);

	/** Initialization. Initializes an instance of the CFloat32Data3DMemory class with pre-allocated memory.
	 * Can only be called by derived classes.
	 *
	 * Initializes an instance of the CFloat32Data3DMemory class. Memory 
	 * is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 * If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 * This function does not set m_bInitialized to true if everything is ok.
	 *
	 * @param _iWidth width of the 2D data (x-axis), must be > 0
	 * @param _iHeight height of the 2D data (y-axis), must be > 0 
	 * @param _iDepth depth of the 2D data (z-axis), must be > 0 
	 * @param _pCustomMemory the custom memory handle
	 */

	bool _initialize(int _iWidth, int _iHeight, int _iDepth, CFloat32CustomMemory* _pCustomMemory);

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CFloat32Data3DMemory();

	/** Destructor. Free allocated memory
	 */
	virtual ~CFloat32Data3DMemory();

	/** Copy the data block pointed to by _pfData to the data block pointed to by m_pfData. 
 	 * The pointer _pfData must point to a block of m_iSize float32s. 
	 *
	 * @param _pfData source data block
	 * @param _iSize total number of data elements, must be equal to the allocated size of the object.
	 */
	void copyData(const float32* _pfData, size_t _iSize);

	/** Set each element of the data to a specified scalar value.
	 *
	 * @param _fScalar scalar value
	 */
	void setData(float32 _fScalar);

	/** Set all data to zero
 	 */
	void clearData();

	/** Get a pointer to the data block, represented as a 1-dimensional
	 * array of float32 values. The data memory is still "owned" by the 
	 * CFloat32Data3DMemory instance; this memory may NEVER be freed by the 
	 * caller of this function. If changes are made to this data, the
	 * function updateStatistics() should be called after completion of
	 * all changes.
	 *
	 * @return pointer to the 1-dimensional 32-bit floating point data block
	 */
	float32* getData();
	
	/** Get a const pointer to the data block, represented as a 1-dimensional
	 * array of float32 values. The data memory is still "owned" by the 
	 * CFloat32Data3DMemory instance; this memory may NEVER be freed by the 
	 * caller of this function. If changes are made to this data, the
	 * function updateStatistics() should be called after completion of
	 * all changes.
	 *
	 * @return pointer to the 1-dimensional 32-bit floating point data block
	 */
	const float32* getDataConst() const;	

	/** Get a float32*** to the data block, represented as a 3-dimensional array of float32 values. 
	 *
	 * After the call p = getData3D(), use p[iz][iy][ix] to access element (ix, iy, iz).
	 * The data memory and pointer array are still "owned" by the CFloat32Data3DMemory 
	 * instance; this memory may NEVER be freed by the caller of this function. 
	 * If changes are made to this data, the function updateStatistics() 
	 * should be called after completion of all changes. 
	 *
	 * @return pointer to the 3-dimensional 32-bit floating point data block
 	 */
	float32*** getData3D();

	/** Get a const float32*** to the data block, represented as a 3-dimensional array of float32 values. 
	 *
	 * After the call p = getData3D(), use p[iy][ix] to access element (ix, iy, iz).
	 * The data memory and pointer array are still "owned" by the CFloat32Data3DMemory 
	 * instance; this memory may NEVER be freed by the caller of this function. 
	 * If changes are made to this data, the function updateStatistics() 
	 * should be called after completion of all changes. 
	 *
	 * @return pointer to the 3-dimensional 32-bit floating point data block
 	 */
	const float32*** getData3DConst() const;

	/** Update data statistics, such as minimum and maximum value, after the data has been modified. 
	 */
	virtual void updateStatistics();

	/** Get the minimum value in the data block.
	 * If the data has been changed after construction, the function
	 * updateStatistics() must be called at least once before 
	 * a query can be made on this value.
	 *
	 * @return minimum value in the data block
	 */
	virtual float32 getGlobalMin() const;

	/** Get the maximum value in the data block
	 * If the data has been changed after construction, the function
	 * updateStatistics() must be called at least once before 
	 * a query can be made on this value.
	 *
	 * @return maximum value in the data block
	 */
	virtual float32 getGlobalMax() const;

	/** which type is this class?
	 *
	 * @return DataType: ASTRA_DATATYPE_FLOAT32_PROJECTION or
	 *					 ASTRA_DATATYPE_FLOAT32_VOLUME
	 */
	virtual EDataType getType() const;

	/**
	 * Clamp data to minimum value
	 *
	 * @param _fMin minimum value
	 * @return l-value
	 */
	virtual CFloat32Data3D& clampMin(float32& _fMin);

	/**
	 * Clamp data to maximum value
	 *
	 * @param _fMax maximum value
	 * @return l-value
	 */
	virtual CFloat32Data3D& clampMax(float32& _fMax);

private:
	CFloat32CustomMemory* m_pCustomMemory;
};


//----------------------------------------------------------------------------------------
// Inline member functions
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Get the type of this object.
inline CFloat32Data3DMemory::EDataType CFloat32Data3DMemory::getType() const
{
	return BASE;
}

//----------------------------------------------------------------------------------------
// Get the minimum value in the data block.
inline float32 CFloat32Data3DMemory::getGlobalMin() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fGlobalMin;
}

//----------------------------------------------------------------------------------------
// Get the maximum value in the data block
inline float32 CFloat32Data3DMemory::getGlobalMax() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fGlobalMax;
}

//----------------------------------------------------------------------------------------
// Get a pointer to the data block, represented as a 1-dimensional array of float32 values.
inline float32* CFloat32Data3DMemory::getData()
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pfData;
}

//----------------------------------------------------------------------------------------
// Get a const pointer to the data block, represented as a 1-dimensional array of float32 values.
inline const float32* CFloat32Data3DMemory::getDataConst() const
{
	ASTRA_ASSERT(m_bInitialized);
	return (const float32*)m_pfData;
}

//----------------------------------------------------------------------------------------
// Get a float32** to the data block, represented as a 3-dimensional array of float32 values.
inline float32*** CFloat32Data3DMemory::getData3D()
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pppfDataSliceInd;
}

//----------------------------------------------------------------------------------------
// Get a const float32** to the data block, represented as a 3-dimensional array of float32 values.
inline const float32*** CFloat32Data3DMemory::getData3DConst() const
{
	ASTRA_ASSERT(m_bInitialized);
	return (const float32***)m_pppfDataSliceInd;
}
//----------------------------------------------------------------------------------------

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32DATA2D
