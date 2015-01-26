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

#ifndef _INC_ASTRA_FLOAT32VOLUMEDATA3DMEMORY
#define _INC_ASTRA_FLOAT32VOLUMEDATA3DMEMORY

#include "Float32Data3DMemory.h"
#include "VolumeGeometry3D.h"
#include "Float32VolumeData3D.h"

namespace astra {

/**
 * This class represents three-dimensional Volume Data where the entire data block is stored in memory.
 */
class _AstraExport CFloat32VolumeData3DMemory : public CFloat32Data3DMemory, public CFloat32VolumeData3D
{
public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CFloat32VolumeData3DMemory();

	/** Constructor. Create an instance of the CFloat32VolumeData3DMemory class without initializing the data.
	 *
	 * Memory is allocated for the data block. The allocated memory is not cleared and 
	 * its contents after construction is undefined. Construction may be followed by a
	 * call to copyData() to fill the memory block.
	 * The size of the data is determined by the specified volume geometry object.
	 *
	 * @param _pGeometry Volume Geometry object. This object will be HARDCOPIED into this class.
	 */
	CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry);

	/** Constructor. Create an instance of the CFloat32VolumeData3DMemory class with initialization of the data.
	 *
	 * Memory is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified volume geometry object.
	 *
	 * @param _pGeometry Volume Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pfData pointer to a one-dimensional float32 data block
	 */
	CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry, const float32* _pfData);

	/** Constructor. Create an instance of the CFloat32VolumeData3DMemory class with scalar initialization of the data.
	 *
	 * Memory is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified volume geometry object.
	 *
	 * @param _pGeometry Volume Geometry object. This object will be HARDCOPIED into this class.
	 * @param _fScalar scalar value
	 */
	CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry, float32 _fScalar);

	/** Constructor. Create an instance of the CFloat32VolumeData3DMemory class with pre-allocated memory.
	 *
	 * Creates an instance of the CFloat32VolumeData3DMemory class. Memory 
	 * is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 *
	 * @param _pGeometry Volume Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pCustomMemory custom memory handle
	 *
	 */
	CFloat32VolumeData3DMemory(CVolumeGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory);

	/** Destructor.
	 */
	virtual ~CFloat32VolumeData3DMemory();

	/** Initialization. Initializes of the CFloat32VolumeData3DMemory class without initializing the data.
	 *
	 * Memory is allocated for the data block. The allocated memory is not cleared and 
	 * its contents after construction is undefined. Construction may be followed by a
	 * call to copyData() to fill the memory block.
	 * The size of the data is determined by the specified volume geometry object.
	 *
	 * @param _pGeometry Volume Geometry of the data. This object will be HARDCOPIED into this class.
	 * @return Initialization of the base class successful.
	 */
	bool initialize(CVolumeGeometry3D* _pGeometry);

	/** Initialization. Initializes an instance of the CFloat32VolumeData3DMemory class with initialization of the data.
	 *
	 * Memory is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified volume geometry object.
	 *
	 * @param _pGeometry Volume Geometry of the data. This object will be HARDCOPIED into this class.
	 * @param _pfData pointer to a one-dimensional float32 data block
	 */
	bool initialize(CVolumeGeometry3D* _pGeometry, const float32* _pfData);

	/** Initialization. Initializes an instance of the CFloat32VolumeData3DMemory class with scalar initialization of the data.
	 *
	 * Memory is allocated for the data block and the contents of the memory pointed to by 
	 * _pfData is copied into the allocated memory. 
	 * The size of the data is determined by the specified volume geometry object.
	 *
	 * @param _pGeometry Volume Geometry of the data. This object will be HARDCOPIED into this class.
	 * @param _fScalar scalar value
	 */
	bool initialize(CVolumeGeometry3D* _pGeometry, float32 _fScalar);

	/** Initialization. Initializes an instance of the CFloat32VolumeData3DMemory class with pre-allocated memory.
	 *
	 * Memory is pre-allocated and passed via the abstract CFloat32CustomMemory handle
	 * class. The handle will be deleted when the memory can be freed.
	 * You should override the destructor to provide custom behaviour on free.
	 *
	 * @param _pGeometry Volume Geometry object.  This object will be HARDCOPIED into this class.
	 * @param _pCustomMemory custom memory handle
	 *
	 */
	bool initialize(CVolumeGeometry3D* _pGeometry, CFloat32CustomMemory* _pCustomMemory);

	/** Which type is this class?
	 *
	 * @return DataType: VOLUME
	 */
	virtual CFloat32Data3D::EDataType getType() const;

	/** Get the volume geometry.
	 *
	 * @return pointer to volume geometry.
	 */
	CVolumeGeometry3D* getGeometry();

	/**
	 * Gets a slice, containing all voxels with a given x (= column) index.
	 */
	CFloat32VolumeData2D * fetchSliceX(int _iColumnIndex) const;

	/**
	 * Gets a slice, containing all voxels with a given y (= row) index.
	 */
	CFloat32VolumeData2D * fetchSliceY(int _iRowIndex) const;

	/**
	 * Gets a slice, containing all voxels with a given z (= slice) index.
	 */
	CFloat32VolumeData2D * fetchSliceZ(int _iSliceIndex) const;

	/**
	 * Gets a slice, containing all voxels with a given x (= column) index.
	 */
	void returnSliceX(int _iColumnIndex, CFloat32VolumeData2D * _pSliceData);

	/**
	 * Gets a slice, containing all voxels with a given y (= row) index.
	 */
	void returnSliceY(int _iRowIndex, CFloat32VolumeData2D * _pSliceData);

	/**
	 * Copies data from a 2D slice containing all voxels with a given z (= slice) index to the
	 * 3D  memory stored in this class.
	 */
	void returnSliceZ(int _iSliceIndex, CFloat32VolumeData2D * _pSliceData);

	/** This SLOW function returns a volume value stored a specific index in the array.
	 *  Reading values in this way might cause a lot of unnecessary memory operations, don't
	 *  use it in time-critical code.
	 * 
	 *  @param _iIndex Index in the array if the data were stored completely in main memory
	 *  @return The value the location specified by _iIndex
	 */
	virtual float32 getVoxelValue(int _iIndex);

	/** This SLOW function stores a voxel value at a specific index in the array.
	 *  Writing values in this way might cause a lot of unnecessary memory operations, don't
	 *  use it in time-critical code.
	 * 
	 *  @param _iIndex Index in the array if the data were stored completely in main memory
	 *  @param _fValue The value to be stored at the location specified by _iIndex
	 */
	virtual void setVoxelValue(int _iIndex, float32 _fValue);

	/**
	 * Overloaded Operator: data = data (pointwise)
	 *
	 * @param _dataIn r-value
	 * @return l-value
	 */
	CFloat32VolumeData3DMemory& operator=(const CFloat32VolumeData3DMemory& _dataIn);
};

//----------------------------------------------------------------------------------------
// Get the projection geometry.
inline CVolumeGeometry3D* CFloat32VolumeData3DMemory::getGeometry()
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pGeometry;
}

//----------------------------------------------------------------------------------------
// Get type
inline CFloat32Data3D::EDataType CFloat32VolumeData3DMemory::getType() const
{
	return VOLUME;
}


} // end namespace astra

#endif // _INC_ASTRA_FLOAT32VOLUMEDATA3DMEMORY
