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

#ifndef _INC_ASTRA_FLOAT32VOLUMEDATA3D
#define _INC_ASTRA_FLOAT32VOLUMEDATA3D

#include "Float32Data3D.h"
#include "Float32VolumeData2D.h"
#include "VolumeGeometry3D.h"

namespace astra {

/**
 * This asbtract class represents three-dimensional Volume Data.
 */
class _AstraExport CFloat32VolumeData3D : public virtual CFloat32Data3D
{
protected:
	CVolumeGeometry3D * m_pGeometry;

public:
	/** Default constructor. 
	 */
	CFloat32VolumeData3D();

	/** Destructor.
	 */
	virtual ~CFloat32VolumeData3D();

	/**
	 * Returns number of rows
	 *
	 * @return number of rows
	 */
	int getRowCount() const;

	/**
	 * Returns number of columns
	 *
	 * @return number of columns
	 */
	int getColCount() const;

	/**
	 * Returns number of slices
	 *
	 * @return number of slices
	 */
	int getSliceCount() const;

	/**
	 * Returns total number of volumes
	 *
	 * @return total number of volumes
	 */
	int getVoxelTotCount() const;

	/** Which type is this class?
	 *
	 * @return DataType: VOLUME
	 */
	virtual CFloat32Data3D::EDataType getType() const;
	
	/** Fetch a slice from the data in the x direction.  Note that if you update the 2D data slice, the data in the 
	 * 3d data object will remain unaltered.  To copy the data you must return the data by calling 'returnSliceX'.
	 * You should not delete data fetched with this function yourself, instead call the 'returnSliceX' function.
	 *
	 * @param _iColumnIndex slice number
	 * @return Volume data object
	 */
	virtual CFloat32VolumeData2D* fetchSliceX(int _iColumnIndex) const = 0;

	/** Fetch a slice from the data in the y direction.  Note that if you update the 2D data slice, the data in the 
	 * 3d data object will remain unaltered.  To copy the data you must return the data by calling 'returnSliceY'.
	 * You should not delete data fetched with this function yourself, instead call the 'returnSliceY' function.
	 *
	 * @param _iRowIndex slice number
	 * @return Volume data object
	 */
	virtual CFloat32VolumeData2D* fetchSliceY(int _iRowIndex) const = 0;

	/** Fetch a slice from the data in the z direction.  Note that if you update the 2D data slice, the data in the 
	 * 3d data object will remain unaltered.  To copy the data you must return the data by calling 'returnSliceZ'.
	 * You should not delete data fetched with this function yourself, instead call the 'returnSliceZ' function.
	 *
	 * @param _iSliceIndex slice number
	 * @return Volume data object
	 */
	virtual CFloat32VolumeData2D* fetchSliceZ(int _iSliceIndex) const = 0;

	/** Return a slice from the data in the x direction to the 3d data.  The data will be deleted. If the slice was 
	 * fetched with 'fetchSliceX', the data will be stored first. 
	 *
	 * @param _iColumnIndex slice number
	 */
	virtual void returnSliceX(int _iColumnIndex, CFloat32VolumeData2D * _pSlice) = 0;
	
	/** Return a slice from the data in the y direction to the 3d data.  The data will be deleted. If the slice was 
	 * fetched with 'fetchSliceY', the data will be stored first. 
	 *
	 * @param _iRowIndex slice number
	 */
	virtual void returnSliceY(int _iRowIndex, CFloat32VolumeData2D * _pSlice) = 0;

	/** Return a slice from the data in the z direction to the 3d data.  The data will be deleted. If the slice was 
	 * fetched with 'fetchSliceZ', the data will be stored first. 
	 *
	 * @param _iSliceIndex slice number
	 */
	virtual void returnSliceZ(int _iSliceIndex, CFloat32VolumeData2D * _pSlice) = 0;

	/** This SLOW function returns a voxel value stored at a specific index in the array.
	 *  Reading values in this way might cause a lot of unnecessary memory operations, don't
	 *  use it in time-critical code.
	 * 
	 *  @param _iIndex Index in the array if the data were stored completely in main memory
	 *  @return The value stored at the location specified by _iIndex
	 */
	virtual float32 getVoxelValue(int _iIndex) = 0;

	/** This SLOW function stores a voxel value at a specific index in the array.
	 *  Writing values in this way might cause a lot of unnecessary memory operations, don't
	 *  use it in time-critical code.
	 * 
	 *  @param _iIndex Index in the array if the data were stored completely in main memory
	 *  @param _fValue The value to be stored at the location specified by _iIndex
	 */
	virtual void setVoxelValue(int _iIndex, float32 _fValue) = 0;

	/**
	 * Overloaded Operator: data += data (pointwise)
	 *
	 * @param _data r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator+=(const CFloat32VolumeData3D& _data);

	/**
	 * Overloaded Operator: data -= data (pointwise)
	 *
	 * @param _data r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator-=(const CFloat32VolumeData3D& _data);

	/**
	 * Overloaded Operator: data *= data (pointwise)
	 *
	 * @param _data r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator*=(const CFloat32VolumeData3D& _data);
	
	/**
	 * Overloaded Operator: data *= scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator*=(const float32& _fScalar);
	
	/**
	 * Overloaded Operator: data /= scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator/=(const float32& _fScalar);

	/**
	 * Overloaded Operator: data += scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator+=(const float32& _fScalar);

	/**
	 * Overloaded Operator: data -= scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32VolumeData3D& operator-=(const float32& _fScalar);

	/**
	 * Gives access to the geometry stored in this class
	 *
	 * @return The geometry describing the data stored in this volume
	 */
	virtual CVolumeGeometry3D* getGeometry() const;

	/** Change the projection geometry.
	 *  Note that this can't change the dimensions of the data.
	 */
	virtual void changeGeometry(CVolumeGeometry3D* pGeometry);
};

//----------------------------------------------------------------------------------------
// get row count
inline int CFloat32VolumeData3D::getRowCount() const
{
	return m_iHeight;
}

//----------------------------------------------------------------------------------------
// get column count
inline int CFloat32VolumeData3D::getColCount() const
{
	return m_iWidth;
}

//----------------------------------------------------------------------------------------
// get slice count
inline int CFloat32VolumeData3D::getSliceCount() const
{
	return m_iDepth;
}

//----------------------------------------------------------------------------------------
// get total voxel count
inline int CFloat32VolumeData3D::getVoxelTotCount() const
{
	return m_iHeight * m_iWidth * m_iDepth;
}

//----------------------------------------------------------------------------------------
// get type
inline CFloat32Data3D::EDataType CFloat32VolumeData3D::getType() const
{
	return CFloat32Data3D::VOLUME;
}

//----------------------------------------------------------------------------------------
// Get the volume geometry.
inline CVolumeGeometry3D* CFloat32VolumeData3D::getGeometry() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pGeometry;
}
//----------------------------------------------------------------------------------------

} // end namespace astra

#endif // _INC_ASTRA_FLOAT32VOLUMEDATA2D
