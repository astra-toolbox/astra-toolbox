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
