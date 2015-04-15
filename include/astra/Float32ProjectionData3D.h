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

#ifndef _INC_ASTRA_FLOAT32PROJECTIONDATA3D
#define _INC_ASTRA_FLOAT32PROJECTIONDATA3D

#include "Float32Data3D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"
#include "ProjectionGeometry3D.h"

namespace astra {

/**
 * This asbtract class represents three-dimensional Projection Data.
 */
class _AstraExport CFloat32ProjectionData3D : public virtual CFloat32Data3D
{
protected:
	/** The projection geometry for this data.
	 */
	CProjectionGeometry3D* m_pGeometry;

public:

	/** Default constructor. 
	 */
	CFloat32ProjectionData3D();

	/** Destructor.
	 */
	virtual ~CFloat32ProjectionData3D();
	
	/** Get the number of detectors in one detector column.
	 *
	 * @return number of detectors
	 */
	int getDetectorRowCount() const;

	/** Get the number of detectors in one detector row.
	 *
	 * @return number of detectors
	 */
	int getDetectorColCount() const;

	/** Get the total number of detectors.
	 *
	 * @return number of detectors
	 */
	int getDetectorTotCount() const;

	/** Get the number of projection angles.
	 *
	 * @return number of projection angles
	 */
	int getAngleCount() const;

	/** Which type is this class?
	 *
	 * @return DataType: ASTRA_DATATYPE_FLOAT32_PROJECTION 
	 */
	virtual CFloat32Data3D::EDataType getType() const;

	/** Fetch a COPY of a projection of the data.  Note that if you update the 2D data slice, the data in the 
	 * 3d data object will remain unaltered.  To copy the data back in the 3D-volume you must return the data by calling 'returnProjection'.
	 *
	 * @param _iProjectionNr projection number
	 * @return Volume data object
	 */
	virtual CFloat32VolumeData2D* fetchProjection(int _iProjectionNr) const = 0;
	
	/** Return a projection slice to the 3d data.  The data will be deleted. If the slice was fetched with 
	 * 'fetchProjection', the data will be stored first. 
	 *
	 * @param _iProjectionNr projection number
	 * @param _pProjection 2D Projection Data
	 */
	virtual void returnProjection(int _iProjectionNr, CFloat32VolumeData2D* _pProjection) = 0;

	/** Fetch a COPY of a sinogram slice of the data.  Note that if you update the 2D data slice, the data in the 
	 * 3d data object will remain unaltered.  To copy the data back in the 3D-volume you must return the data by calling 'returnSlice'.
	 *
	 * @param _iSliceNr slice number
	 * @return Sinogram data object
	 */
	virtual CFloat32ProjectionData2D* fetchSinogram(int _iSliceNr) const = 0;

	/** Return a sinogram slice to the 3d data.  The data will be stored in the 3D Data object.
	 *
	 * @param _iSliceNr slice number
	 * @param _pSinogram2D 2D Sinogram Object.
	 */
	virtual void returnSinogram(int _iSliceNr, CFloat32ProjectionData2D* _pSinogram2D) = 0;

	/** This SLOW function returns a detector value stored a specific index in the array.
	 *  Reading values in this way might cause a lot of unnecessar__y memory operations, don't
	 *  use it in time-critical code.
	 * 
	 *  @param _iIndex Index in the array if the data were stored completely in main memory
	 *  @return The value the location specified by _iIndex
	 */
	virtual float32 getDetectorValue(int _iIndex) = 0;

	/** This SLOW function stores a detector value at a specific index in the array.
	 *  Writing values in this way might cause a lot of unnecessary memory operations, don't
	 *  use it in time-critical code.
	 * 
	 *  @param _iIndex Index in the array if the data were stored completely in main memory
	 *  @param _fValue The value to be stored at the location specified by _iIndex
	 */
	virtual void setDetectorValue(int _iIndex, float32 _fValue) = 0;

	/**
	 * Overloaded Operator: data += data (pointwise)
	 *
	 * @param _data r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator+=(const CFloat32ProjectionData3D& _data);

	/**
	 * Overloaded Operator: data -= data (pointwise)
	 *
	 * @param _data r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator-=(const CFloat32ProjectionData3D& _data);

	/**
	 * Overloaded Operator: data *= data (pointwise)
	 *
	 * @param _data r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator*=(const CFloat32ProjectionData3D& _data);
	
	/**
	 * Overloaded Operator: data *= scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator*=(const float32& _fScalar);
	
	/**
	 * Overloaded Operator: data /= scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator/=(const float32& _fScalar);

	/**
	 * Overloaded Operator: data += scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator+=(const float32& _fScalar);

	/**
	 * Overloaded Operator: data -= scalar (pointwise)
	 *
	 * @param _fScalar r-value
	 * @return l-value
	 */
	CFloat32ProjectionData3D& operator-=(const float32& _fScalar);

	/** Get the projection geometry.
	 *
	 * @return pointer to projection geometry.
	 */
	virtual CProjectionGeometry3D* getGeometry() const;

	/** Change the projection geometry.
	 *  Note that this can't change the dimensions of the data.
	 */
	virtual void changeGeometry(CProjectionGeometry3D* pGeometry);
};


//----------------------------------------------------------------------------------------
// Inline member functions
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CFloat32ProjectionData3D::getDetectorColCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iWidth;
}

//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CFloat32ProjectionData3D::getDetectorRowCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iDepth;
}

//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CFloat32ProjectionData3D::getDetectorTotCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iWidth * m_iDepth;
}

//----------------------------------------------------------------------------------------
// Get the number of projection angles.
inline int CFloat32ProjectionData3D::getAngleCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iHeight;
}

//----------------------------------------------------------------------------------------
// Get type
inline CFloat32Data3D::EDataType CFloat32ProjectionData3D::getType() const 
{
	return PROJECTION;
}
//----------------------------------------------------------------------------------------
// Get the projection geometry.
inline CProjectionGeometry3D* CFloat32ProjectionData3D::getGeometry() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pGeometry;
}
//----------------------------------------------------------------------------------------


} // end namespace astra

#endif // _INC_ASTRA_FLOAT32PROJECTIONDATA3D
