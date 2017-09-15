/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://www.astra-toolbox.com/

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
*/

#ifndef _INC_ASTRA_FLOAT32PROJECTIONDATA3DMEMORY
#define _INC_ASTRA_FLOAT32PROJECTIONDATA3DMEMORY

#include "Float32Data3DMemory.h"
#include "Float32ProjectionData3D.h"
#include "ParallelProjectionGeometry2D.h" // TEMP

namespace astra {

/**
 * This class represents three-dimensional Projection Data.
 *
 * It contains member functions for accessing this data and for performing 
 * elementary computations on the data.
 * The data block is "owned" by the class, meaning that the class is 
 * responsible for deallocation of the memory involved. 
 *
 * The projection data is stored as a series of consecutive rows, where
 * each row contains the data for a single projection. 
 */
class _AstraExport CFloat32ProjectionData3DMemory : public CFloat32Data3DMemory, public CFloat32ProjectionData3D {

public:

	/** 
	 * Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CFloat32ProjectionData3DMemory();

	/**
	 * Constructor. Create an instance of the CFloat32ProjectionData3DMemory class without initializing the data.
	 *
	 * Memory is allocated for the data block. The allocated memory is not cleared and 
	 * its contents after construction is undefined. Construction may be followed by a
	 * call to copyData() to fill the memory block.
	 * The size of the data is determined by the specified projection geometry object.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 */
	CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry);

	/**
	 * Constructor. Create an instance of the CFloat32ProjectionData3DMemory class with initialization of the data.
	 *
	 * Creates an instance of the CFloat32ProjectionData3DMemory class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified projection geometry object.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pfData pointer to a one-dimensional float32 data block
	 */
	CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry, float32* _pfData);

	/**
	 * Constructor. Create an instance of the CFloat32ProjectionData3DMemory class filled with scalar data.
	 *
	 * Creates an instance of the CFloat32ProjectionData3DMemory class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified projection geometry object.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _fScalar scalar data
	 */
	CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry, float32 _fScalar);

	/** Constructor. Create an instance of the CFloat32ProjectionData3DMemory class with pre-allocated memory.
	 *
	 * Creates an instance of the CFloat32ProjectionData3DMemory class. Memory 
	 * is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pCustomMemory custom memory handle
	 *
	 */
	CFloat32ProjectionData3DMemory(CProjectionGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory);

	/** 
	 * Destructor.
	 */
	virtual ~CFloat32ProjectionData3DMemory();
	
	/** Initialization. Initializes an instance of the CFloat32ProjectionData3DMemory class, without filling the data block.
	 *
	 * Initializes an instance of the CFloat32ProjectionData3DMemory class. Memory is allocated for the 
	 * data block. The allocated memory is not cleared and its contents after 
	 * construction is undefined. Initialization may be followed by a call to 
	 * copyData() to fill the memory block. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 * @param _pGeometry Projection Geometry of the data. This object will be HARDCOPIED into this class.
	 * @return Initialization of the base class successfull.
	 */
	bool initialize(CProjectionGeometry3D* _pGeometry);

	/** Initialization. Initializes an instance of the CFloat32ProjectionData3DMemory class with scalar initialization. 
	 *
	 * Initializes an instance of the CFloat32ProjectionData3DMemory class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 * @param _pGeometry Projection Geometry of the data. This object will be HARDCOPIED into this class.
	 * @param _fScalar scalar value
	 */
	bool initialize(CProjectionGeometry3D* _pGeometry, float32 _fScalar);

	/** Initialization. Initializes an instance of the CFloat32ProjectionData3DMemory class with initialization of the data block. 
	 *
	 * Initializes an instance of the CFloat32ProjectionData3DMemory class. Memory 
	 * is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. If the object has been initialized before, the 
	 * object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 * @param _pGeometry Projection Geometry of the data. This object will be HARDCOPIED into this class.
	 * @param _pfData pointer to a one-dimensional float32 data block
	 */
	bool initialize(CProjectionGeometry3D* _pGeometry, const float32* _pfData);

	/** Initialization. Initializes an instance of the CFloat32ProjectionData3DMemory class with pre-allocated memory.
	 *
	 * Memory is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 *
	 * @param _pGeometry Projection Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pCustomMemory custom memory handle
	 *
	 */
	bool initialize(CProjectionGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory);



	/** Which type is this class?
	 *
	 * @return DataType: PROJECTION 
	 */
	virtual EDataType getType() const;

	/**
	 * Overloaded Operator: data = data (pointwise)
	 *
	 * @param _dataIn r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3DMemory& operator=(const CFloat32ProjectionData3DMemory& _dataIn);
};
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
// Get type.
inline CFloat32Data3D::EDataType CFloat32ProjectionData3DMemory::getType() const
{
	return PROJECTION;
}
//----------------------------------------------------------------------------------------

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32PROJECTIONDATA3DMEMORY
