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

#ifndef _INC_ASTRA_PROJECTIONGEOMETRY3D
#define _INC_ASTRA_PROJECTIONGEOMETRY3D

#include "Globals.h"
#include "Config.h"
#include "Vector3D.h"

#include <string>
#include <cmath>
#include <vector>

namespace astra
{

class XMLNode;

/**
 * This class defines the interface for each 3D projection geometry. 
 * It has a number of data fields, such as width and height of detector
 * pixels, projection angles and number of rows and columns of detector pixels.
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorRowCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorColCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorWidth, float, Width of each detector.}
 * \astra_xml_item{DetectorHeight, float, Width of each detector.}
 * \astra_xml_item{ProjectionAngles, vector of float, projection angles in radians.}
 */
class _AstraExport CProjectionGeometry3D
{

protected:

	/** Has the object been intialized with acceptable values?
	 */
	bool m_bInitialized;

	/** Number of projection angles.
	 */
	int m_iProjectionAngleCount;

	/** Number of rows of detectors.
	 */
	int m_iDetectorRowCount;

	/** Number of columns of detectors.
	 */
	int m_iDetectorColCount;

	/** Total number of detectors.
	 */
	int m_iDetectorTotCount;

	/** The x-distance between projected rays on the detector plate (or width of projected strips).
	 */
	float32 m_fDetectorSpacingX;

	/** The y-distance between projected rays on the detector plate (or height of projected strips).
	 */
	float32 m_fDetectorSpacingY;

	/** Dynamically allocated array of projection angles. All angles are represented in radians and lie in 
	 * the [0,2pi[ interval.
	 */
	float32* m_pfProjectionAngles;
	
	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling initialize() 
	 * is not allowed, except calling the member function isInitialized().
	 *
	 */
	CProjectionGeometry3D();

	/** Constructor. Create an instance of the CProjectionGeometry3D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorRowCount Number of rows of detectors.
	 *  @param _iDetectorColCount Number of columns detectors.
	 *  @param _fDetectorSpacingX Spacing between the detector points on the X-axis, in unit lengths. Assumed to be constant throughout the entire detector plate.
	 *  @param _fDetectorSpacingY Spacing between the detector points on the Y-axis, in unit lengths. Assumed to be constant throughout the entire detector plate.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. All angles 
	 *                             are represented in radians and lie in the [0,2pi[ interval.
	 */
	CProjectionGeometry3D(int _iProjectionAngleCount,
						  int _iDetectorRowCount,
						  int _iDetectorColCount,
						  float32 _fDetectorSpacingX,
						  float32 _fDetectorSpacingY,
						  const float32* _pfProjectionAngles);

	/** Copy constructor. 
	 */
	CProjectionGeometry3D(const CProjectionGeometry3D& _projGeom);

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - number of rows and columns is larger than zero
	 * - detector spacing is larger than zero
	 * - number of angles is larger than zero
	 * - (autofix) each angle lies in [0,2pi[
	 */
	bool _check();
	
	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 * Should only be used by constructors.  Otherwise use the clear() function.
	 */
	void _clear();

	/** Initialize the geometry. If the object has been initialized before, the object is reinitialized 
	 * and memory is freed and reallocated if necessary.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorRowCount Number of rows of detectors.
	 *  @param _iDetectorColCount Number of columns detectors.
	 *  @param _fDetectorSpacingX Spacing between the detector points on the X-axis, in unit lengths. Assumed to be constant throughout the entire detector plate.
	 *  @param _fDetectorSpacingY Spacing between the detector points on the Y-axis, in unit lengths. Assumed to be constant throughout the entire detector plate.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. All angles 
	 *                             are represented in radians and lie in the [0,2pi[ interval.
	 */
	bool _initialize(int _iProjectionAngleCount,
					 int _iDetectorRowCount,
					 int _iDetectorColCount,
					 float32 _fDetectorSpacingX,
					 float32 _fDetectorSpacingY,
					 const float32* _pfProjectionAngles);

public:

	/** Destructor 
	 */
	virtual ~CProjectionGeometry3D();
	
	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 */
	virtual void clear();

	/** Create a hard copy. 
	*/
	virtual CProjectionGeometry3D* clone() const = 0;

	/** Initialize the geometry with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

    /** Get the initialization state of the object.
	 *
	 * @return true iff the object has been initialized
	 */
	bool isInitialized() const;

    /** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(const CProjectionGeometry3D *) const = 0;

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const = 0;

	/** Get the number of projections.
	 *
	 * @return Number of projections
	 */
	int getProjectionCount() const;

	/** Get the number of rows of detectors.
	 *
	 * @return Number of rows of detectors.
	 */
	int getDetectorRowCount() const;
	
	/** Set the number of rows of detectors.
	 *
	 */
	void setDetectorRowCount(const int);

	/** Get the number of columns of detectors.
	 *
	 * @return Number of columns of detectors.
	 */
	int getDetectorColCount() const;

	/** Get the total number of detectors.
	 *
	 * @return Total number of detectors.
	 */
	int getDetectorTotCount() const;

	/** Get the width of a detector.
	 *
	 * @return Width of a detector, in unit lengths
	 */
	float32 getDetectorSpacingX() const;

	/** Get the height of a detector.
	 *
	 * @return Height of a detector, in unit lengths
	 */
	float32 getDetectorSpacingY() const;

	/** Get a projection angle, given by its index. The angle is represented in Radians.
	 *
	 * @return Projection angle with index _iProjectionIndex
	 */
	float32 getProjectionAngle(int _iProjectionIndex) const;

	/** Get a projection angle, given by its index. The angle is represented in degrees.
	 *
	 * @return Projection angle with index _iProjectionIndex
	 */
//	float32 getProjectionAngleDegrees(int _iProjectionIndex) const;

	/** Returns a buffer containing all projection angles. The element count of the buffer is equal
	 *  to the number given by getProjectionAngleCount.
	 *
	 *  The angles are in radians.
	 *
	 * @return Pointer to buffer containing the angles.
	 */
	const float32* getProjectionAngles() const;

	/** Get the column index coordinate of a point on a detector array.
	 *
	 * @param _fOffsetX	Distance between the center of the detector array and a certain point (both on the X-axis).
	 * @return The location of the point in index X-coordinates (still float, not rounded)
	 */
	virtual float32 detectorOffsetXToColIndexFloat(float32 _fOffsetX) const;

	/** Get the row index coordinate of a point on a detector array.
	 *
	 * @param _fOffsetY	Distance between the center of the detector array and a certain point (both on the Y-axis).
	 * @return The location of the point in index Y-coordinates (still float, not rounded)
	 */
	virtual float32 detectorOffsetYToRowIndexFloat(float32 _fOffsetY) const;

	/** Get the offset of a detector on the X-axis based on its index coordinate.
	 *
	 * @param _iIndex	the index of the detector.
	 * @return			the offset from the center of the detector array on the X-axis.
	 */
	virtual float32 indexToDetectorOffsetX(int _iIndex) const;

	/** Get the offset of a detector on the Y-axis based on its index coordinate.
	 *
	 * @param _iIndex	the index of the detector.
	 * @return			the offset from the center of the detector array on the Y-axis.
	 */
	virtual float32 indexToDetectorOffsetY(int _iIndex) const;

	/** Get the offset of a detector on the X-axis based on its column index coordinate.
	 *
	 * @param _iIndex	the index of the detector.
	 * @return			the offset from the center of the detector array on the X-axis.
	 */
	virtual float32 colIndexToDetectorOffsetX(int _iIndex) const;

	/** Get the offset of a detector on the Y-axis based on its row index coordinate.
	 *
	 * @param _iIndex	the index of the detector.
	 * @return			the offset from the center of the detector array on the Y-axis.
	 */
	virtual float32 rowIndexToDetectorOffsetY(int _iIndex) const;
	
	/** Get the row and column index of a detector based on its index.
	 *
	 * @param _iDetectorIndex	in: the index of the detector.
	 * @param _iDetectorRow		out: the row index of the detector.
	 * @param _iDetectorCol		out: the column index of the detector.
	 */
	virtual void detectorIndexToRowCol(int _iDetectorIndex, int& _iDetectorRow, int& _iDetectorCol) const;

	/** Get the angle and detector index of a detector 
	 *
	 * @param _iIndex	the index of the detector.
	 * @param _iAngleIndex	output: index of angle
	 * @param _iDetectorIndex output: index of detector
	 */
	virtual void indexToAngleDetectorIndex(int _iIndex, int& _iAngleIndex, int& _iDetectorIndex) const;

	/** Project a point onto the detector. The 3D point coordinates
	 * are in units. The output fU,fV are the (unrounded) indices of the
	 * detector column and row.
	 * This may fall outside of the actual detector.
	 *
	 * @param fX,fY,fZ	coordinates of the point to project
	 * @param iAngleIndex	the index of the angle to use
	 * @param fU,fV		the projected point.
	 */
	virtual void projectPoint(double fX, double fY, double fZ,
	                          int iAngleIndex,
	                          double &fU, double &fV) const = 0;

	/* Backproject a point onto a plane parallel to a coordinate plane.
	 * The 2D point coordinates are the (unrounded) indices of the detector
	 * column and row. The output is in 3D coordinates in units.
	 * are in units. The output fU,fV are the (unrounded) indices of the
	 * detector column and row.
	 * This may fall outside of the actual detector.
	 */
	virtual void backprojectPointX(int iAngleIndex, double fU, double fV,
	                               double fX, double &fY, double &fZ) const = 0;
	virtual void backprojectPointY(int iAngleIndex, double fU, double fV,
	                               double fY, double &fX, double &fZ) const = 0;
	virtual void backprojectPointZ(int iAngleIndex, double fU, double fV,
	                               double fZ, double &fX, double &fY) const = 0;


	/** Returns true if the type of geometry defined in this class is the one specified in _sType.
	 *
	 * @param _sType geometry type to compare to.
	 * @return true if the type of geometry defined in this class is the one specified in _sType. 
	 */
	 virtual bool isOfType(const std::string& _sType) const = 0;

	 /**
	  * Returns a vector giving the projection direction for a projection and detector index
	  */
	 virtual CVector3D getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex) const = 0;


	//< For Config unused argument checking
	ConfigCheckData* configCheckData;
	friend class ConfigStackCheck<CProjectionGeometry3D>;
};



//----------------------------------------------------------------------------------------
// Inline member functions
//----------------------------------------------------------------------------------------
// Get the initialization state.
inline bool CProjectionGeometry3D::isInitialized() const
{
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CProjectionGeometry3D::getDetectorRowCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iDetectorRowCount;
}

// Set the number of detector rows.
inline void CProjectionGeometry3D::setDetectorRowCount(const int nRows)
{
	ASTRA_ASSERT(m_bInitialized);
	m_iDetectorRowCount = nRows;
}

//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CProjectionGeometry3D::getDetectorColCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iDetectorColCount;
}

//----------------------------------------------------------------------------------------
// Get the number of detectors.
inline int CProjectionGeometry3D::getDetectorTotCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iDetectorTotCount;
}

//----------------------------------------------------------------------------------------
// Get the width of a single detector (in unit lengths).
inline float32 CProjectionGeometry3D::getDetectorSpacingX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fDetectorSpacingX;
}
//----------------------------------------------------------------------------------------
// Get the width of a single detector (in unit lengths).
inline float32 CProjectionGeometry3D::getDetectorSpacingY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fDetectorSpacingY;
}

//----------------------------------------------------------------------------------------
// Get the number of projection angles.
inline int CProjectionGeometry3D::getProjectionCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iProjectionAngleCount;
}

//----------------------------------------------------------------------------------------
// Get a projection angle, represented in Radians.
inline float32 CProjectionGeometry3D::getProjectionAngle(int _iProjectionIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iProjectionIndex >= 0);
	ASTRA_ASSERT(_iProjectionIndex < m_iProjectionAngleCount);

	return m_pfProjectionAngles[_iProjectionIndex];
}

/*
//----------------------------------------------------------------------------------------
// Get a projection angle, represented in degrees.
inline float32 CProjectionGeometry3D::getProjectionAngleDegrees(int _iProjectionIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iProjectionIndex >= 0);
	ASTRA_ASSERT(_iProjectionIndex < m_iProjectionAngleCount);

	return (m_pfProjectionAngles[_iProjectionIndex] * 180.0f / PI32);
}
*/


//----------------------------------------------------------------------------------------
// Get pointer to buffer used to store projection angles.
inline const float32* CProjectionGeometry3D::getProjectionAngles() const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);

	return m_pfProjectionAngles;
}


//----------------------------------------------------------------------------------------
// detector offset X -> detector column index (float)
inline float32 CProjectionGeometry3D::detectorOffsetXToColIndexFloat(float32 _fOffsetX) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);

	return (_fOffsetX / m_fDetectorSpacingX) + ((m_iDetectorColCount-1.0f) / 2.0f);
}

//----------------------------------------------------------------------------------------
// detector offset Y -> detector row index (float)
inline float32 CProjectionGeometry3D::detectorOffsetYToRowIndexFloat(float32 _fOffsetY) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);

	return (_fOffsetY / m_fDetectorSpacingY) + ((m_iDetectorRowCount-1.0f) / 2.0f);
}

//----------------------------------------------------------------------------------------
// detector index -> detector offset X
inline float32 CProjectionGeometry3D::indexToDetectorOffsetX(int _iIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iIndex >= 0);
	ASTRA_ASSERT(_iIndex < m_iDetectorTotCount);

	_iIndex = _iIndex % m_iDetectorColCount;
	return (_iIndex - (m_iDetectorColCount-1.0f) / 2.0f) * m_fDetectorSpacingX;
}

//----------------------------------------------------------------------------------------
// detector index -> detector offset Y
inline float32 CProjectionGeometry3D::indexToDetectorOffsetY(int _iIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iIndex >= 0);
	ASTRA_ASSERT(_iIndex < m_iDetectorTotCount);

	_iIndex = _iIndex / m_iDetectorColCount;
	return -(_iIndex - (m_iDetectorRowCount-1.0f) / 2.0f) * m_fDetectorSpacingY;
}

//----------------------------------------------------------------------------------------
// detector index -> detector offset X
inline float32 CProjectionGeometry3D::colIndexToDetectorOffsetX(int _iIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iIndex >= 0);
	ASTRA_ASSERT(_iIndex < m_iDetectorColCount);

	return (_iIndex - (m_iDetectorColCount-1.0f) / 2.0f) * m_fDetectorSpacingX;
}

//----------------------------------------------------------------------------------------
// detector index -> detector offset Y
inline float32 CProjectionGeometry3D::rowIndexToDetectorOffsetY(int _iIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iIndex >= 0);
	ASTRA_ASSERT(_iIndex < m_iDetectorRowCount);

	return (_iIndex - (m_iDetectorRowCount-1.0f) / 2.0f) * m_fDetectorSpacingY;
}

//----------------------------------------------------------------------------------------
// detector index -> row index & column index
inline void CProjectionGeometry3D::detectorIndexToRowCol(int _iDetectorIndex, int& _iDetectorRow, int& _iDetectorCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iDetectorIndex >= 0);
	ASTRA_ASSERT(_iDetectorIndex < m_iDetectorTotCount);
	
	_iDetectorRow = _iDetectorIndex / m_iDetectorColCount;
	_iDetectorCol = _iDetectorIndex % m_iDetectorColCount;
}

//----------------------------------------------------------------------------------------
inline void CProjectionGeometry3D::indexToAngleDetectorIndex(int _iIndex, int& _iAngleIndex, int& _iDetectorIndex) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iIndex >= 0);
	ASTRA_ASSERT(_iIndex < m_iDetectorTotCount * m_iProjectionAngleCount);

//	int det_row = _iIndex / (m_iDetectorColCount*m_iProjectionAngleCount);
//	int det_col = _iIndex % m_iDetectorColCount;
//
//	_iAngleIndex = _iIndex % (m_iDetectorColCount*m_iProjectionAngleCount) / m_iDetectorColCount;
//	_iDetectorIndex = det_row * m_iDetectorColCount + det_col;

	_iAngleIndex = (_iIndex % (m_iDetectorColCount*m_iProjectionAngleCount)) / m_iDetectorColCount;
	_iDetectorIndex = _iIndex / m_iProjectionAngleCount + (_iIndex % m_iDetectorColCount);

}

//----------------------------------------------------------------------------------------

} // end namespace astra

#endif /* _INC_ASTRA_PROJECTIONGEOMETRY2D */
