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

#ifndef _INC_ASTRA_FLOAT32PROJECTIONDATA2D
#define _INC_ASTRA_FLOAT32PROJECTIONDATA2D

#include "Float32Data2D.h"
#include "ProjectionGeometry2D.h"

namespace astra {

/**
 * This class represents two-dimensional Projection Data.
 *
 * It contains member functions for accessing this data and for performing 
 * elementary computations on the data.
 * The data block is "owned" by the class, meaning that the class is 
 * responsible for deallocation of the memory involved. 
 *
 * The projection data is stored as a series of consecutive rows, where
 * each row contains the data for a single projection. 
 */
class _AstraExport CFloat32ProjectionData2D : public CFloat32Data2D {

public:

	/** 
	 * Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CFloat32ProjectionData2D();

	/**
	 * Constructor. Create an instance of the CFloat32ProjectionData2D class without initializing the data.
	 *
	 * Memory is allocated for the data block. The allocated memory is not cleared and 
	 * its contents after construction is undefined. Construction may be followed by a
	 * call to copyData() to fill the memory block.
	 * The size of the data is determined by the specified projection geometry object.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 */
	CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry);

	/**
	 * Constructor. Create an instance of the CFloat32ProjectionData2D class with initialization of the data.
	 *
	 * Creates an instance of the CFloat32ProjectionData2D class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified projection geometry object.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pfData pointer to a one-dimensional float32 data block
	 */
	CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry, float32* _pfData);

	/**
	 * Constructor. Create an instance of the CFloat32ProjectionData2D class with initialization of the data.
	 *
	 * Creates an instance of the CFloat32ProjectionData2D class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified projection geometry object.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _fScalar scalar value to be put at each index.
	 */
	CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry, float32 _fScalar);

	/**
	 * Copy constructor
	 */
	CFloat32ProjectionData2D(const CFloat32ProjectionData2D& _other);
	
	/** Constructor. Create an instance of the CFloat32ProjectionData2D class with pre-allocated memory.
	 *
	 * Creates an instance of the CFloat32ProjectionData2D class. Memory 
	 * is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pCustomMemory custom memory handle
	 *
	 */
	CFloat32ProjectionData2D(CProjectionGeometry2D* _pGeometry, CFloat32CustomMemory* _pCustomMemory);

	/**
	 * Assignment operator
	 */
	CFloat32ProjectionData2D& operator=(const CFloat32ProjectionData2D& _other);

	/** 
	 * Destructor.
	 */
	virtual ~CFloat32ProjectionData2D();
	
	/** Initialization. Initializes an instance of the CFloat32ProjectionData2D class, without filling the data block.
	 *
	 * Initializes an instance of the CFloat32Data2D class. Memory is allocated for the 
	 * data block. The allocated memory is not cleared and its contents after 
	 * construction is undefined. Initialization may be followed by a call to 
	 * copyData() to fill the memory block. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 * @param _pGeometry Projection Geometry of the data. This object will be HARDCOPIED into this class.
	 * @return Initialization of the base class successfull.
	 */
	bool initialize(CProjectionGeometry2D* _pGeometry);

	/** Initialization. Initializes an instance of the CFloat32Data2D class with initialization of the data block. 
	 *
	 * Initializes an instance of the CFloat32Data2D class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 * @param _pGeometry Projection Geometry of the data. This object will be HARDCOPIED into this class.
	 * @param _pfData pointer to a one-dimensional float32 data block
	 */
	bool initialize(CProjectionGeometry2D* _pGeometry, const float32* _pfData);

	/** Initialization. Initializes an instance of the CFloat32Data2D class with initialization of the data block. 
	 *
	 * Initializes an instance of the CFloat32Data2D class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 * @param _pGeometry Projection Geometry of the data. This object will be HARDCOPIED into this class.
	 * @param _fScalar scalar value to be put at each index.
	 */
	bool initialize(CProjectionGeometry2D* _pGeometry, float32 _fScalar);
	
	/** Initialization. Initializes an instance of the CFloat32ProjectionData2D class with pre-allocated memory.
	 *
	 * Memory is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pCustomMemory custom memory handle
	 *
	 */
	bool initialize(CProjectionGeometry2D* _pGeometry, CFloat32CustomMemory* _pCustomMemory);

	/** Get the number of detectors.
	 *
	 * @return number of detectors
	 */
	int getDetectorCount() const;

	/** Get the number of projection angles.
	 *
	 * @return number of projection angles
	 */
	int getAngleCount() const;

	/** Get a pointer to the data of a single projection angle.
	 *
	 * The data memory is still "owned" by the 
	 * CFloat32ProjectionData2D instance; this memory may NEVER be freed by the 
	 * caller of this function. If changes are made to this data, the
	 * function updateStatistics() should be called after completion of
	 * all changes.
	 *
	 * @return pointer to the data
	 */
	float32* getSingleProjectionData(int _iAngleIndex);

	/** Get a const pointer to the data of a single projection angle.
	 *
	 * The data memory is still "owned" by the 
	 * CFloat32ProjectionData2D instance; this memory may NEVER be freed by the 
	 * caller of this function. 
	 *
	 * @return pointer to the data
	 */
	const float32* getSingleProjectionDataConst(int _iAngleIndex) const;

	/** Which type is this class?
	 *
	 * @return DataType: PROJECTION 
	 */
	virtual EDataType getType() const;

	/** Get the projection geometry.
	 *
	 * @return pointer to projection geometry.
	 */
	virtual CProjectionGeometry2D* getGeometry() const;

	/** Change the projection geometry.
 	 *  Note that this can't change the dimensions of the data.
	 */
	virtual void changeGeometry(CProjectionGeometry2D* pGeometry);

protected:

	/** The projection geometry for this data.
	 */
	CProjectionGeometry2D* m_pGeometry;

};
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CFloat32ProjectionData2D::getDetectorCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iWidth;
}

//----------------------------------------------------------------------------------------
// Get the number of projection angles.
inline int CFloat32ProjectionData2D::getAngleCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iHeight;
}

//----------------------------------------------------------------------------------------
// Get the projection geometry.
inline CProjectionGeometry2D* CFloat32ProjectionData2D::getGeometry() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pGeometry;
}

//----------------------------------------------------------------------------------------
// Get type.
inline CFloat32Data2D::EDataType CFloat32ProjectionData2D::getType() const
{
	return PROJECTION;
}
//----------------------------------------------------------------------------------------

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32PROJECTIONDATA2D
