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

#ifndef _INC_ASTRA_PROJECTIONGEOMETRY2D
#define _INC_ASTRA_PROJECTIONGEOMETRY2D

#include "Globals.h"
#include "Config.h"
#include "Vector3D.h"

#include <string>
#include <cmath>
#include <vector>

namespace astra
{

/**
 * This abstract base class defines the projection geometry. 
 * It has a number of data fields, such as width of detector
 * pixels, projection angles, number of detector pixels and object offsets
 * for every projection angle.
 */
class _AstraExport CProjectionGeometry2D
{

protected:

	bool m_bInitialized;	///< Has the object been intialized?

	/** Number of projection angles
	 */
	int m_iProjectionAngleCount;

	/** Number of detectors, i.e., the number of detector measurements for each projection angle.
	 */
	int m_iDetectorCount;

	/** Width of a detector pixel, i.e., the distance between projected rays (or width of projected strips).
	 */
	float32 m_fDetectorWidth;

	/** An array of m_iProjectionAngleCount elements containing an extra detector offset for each projection.
	 */
	float32* m_pfExtraDetectorOffset;

	/** Dynamically allocated array of projection angles. All angles are represented in radians and lie in 
	 * the [0,2pi[ interval.
	 */
	float32* m_pfProjectionAngles;

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not 
	 * allowed, except calling the member function isInitialized().
	 *
	 */
	CProjectionGeometry2D();

	/** Constructor. Create an instance of the CProjectionGeometry2D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 *  @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. 
	 *  All angles are represented in radians.
	 */
	CProjectionGeometry2D(int _iProjectionAngleCount, 
						  int _iDetectorCount, 
						  float32 _fDetectorWidth, 
						  const float32* _pfProjectionAngles,
						  const float32* _pfExtraDetectorOffsets = 0);

	/** Copy constructor. 
	 */
	CProjectionGeometry2D(const CProjectionGeometry2D& _projGeom);

	/** Check variable values.
	 */
	bool _check();
	
	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 * Should only be used by constructors.  Otherwise use the clear() function.
	 */
	void _clear();

	/** Initialization. Initializes an instance of the CProjectionGeometry2D class. If the object has been 
	 * initialized before, the object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 *  @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 */
	bool _initialize(int _iProjectionAngleCount,
					 int _iDetectorCount,
					 float32 _fDetectorWidth,
					 const float32* _pfProjectionAngles,
					 const float32* _pfExtraDetectorOffsets = 0);

public:

	/** Destructor 
	 */
	virtual ~CProjectionGeometry2D();

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 */
	virtual void clear();

	/** Create a hard copy. 
	*/
	virtual CProjectionGeometry2D* clone() = 0;

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
	virtual bool isEqual(CProjectionGeometry2D*) const = 0;

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const = 0;

	/** Get the number of projection angles.
	 *
	 * @return Number of projection angles
	 */
	int getProjectionAngleCount() const;

	/** Get the number of detectors.
	 *
	 * @return Number of detectors, i.e., the number of detector measurements for each projection angle.
	 */
	int getDetectorCount() const;

	/** Get the width of a detector.
	 *
	 * @return Width of a detector, in unit lengths
	 */
	float32 getDetectorWidth() const;

	/** Get a projection angle, given by its index. The angle is represented in Radians.
	 *
	 * @return Projection angle with index _iProjectionIndex
	 */
	float32 getProjectionAngle(int _iProjectionIndex) const;

	/** Returns a buffer containing all projection angles. The element count of the buffer is equal
	 *  to the number given by getProjectionAngleCount.
	 *
	 *  The angles are in radians.
	 *
	 * @return Pointer to buffer containing the angles.
	 */
	const float32* getProjectionAngles() const;

	/** Get a projection angle, given by its index. The angle is represented in degrees.
	 *
	 * @return Projection angle with index _iProjectionIndex
	 */
	float32 getProjectionAngleDegrees(int _iProjectionIndex) const;

	float32 getExtraDetectorOffset(int iAngle) const;
	const float32* getExtraDetectorOffset() const { return m_pfExtraDetectorOffset; }

	/** Get the index coordinate of a point on a detector array.
	 *
	 * @param _fOffset	distance between the center of the detector array and a certain point
	 * @return			the location of the point in index coordinates (still float, not rounded) 
	 */
	virtual float32 detectorOffsetToIndexFloat(float32 _fOffset) const;

	/** Get the index coordinate of a point on a detector array.
	 *
	 * @param _fOffset	distance between the center of the detector array and a certain point
	 * @return			the index of the detector that is hit, -1 if detector array isn't hit.
	 */
	virtual int detectorOffsetToIndex(float32 _fOffset) const;

	/** Get the offset of a detector based on its index coordinate.
	 *
	 * @param _iIndex	the index of the detector.
	 * @return			the offset from the center of the detector array.
	 */
	virtual float32 indexToDetectorOffset(int _iIndex) const;

	/** Get the angle and detector index of a sinogram pixel
	 *
	 * @param _iIndex	the index of the detector pixel in the sinogram.
	 * @param _iAngleIndex	output: index of angle
	 * @param _iDetectorIndex output: index of detector
	 */
	virtual void indexToAngleDetectorIndex(int _iIndex, int& _iAngleIndex, int& _iDetectorIndex) const;

	/** Get the value for t and theta, based upon the row and column index.
	 *
	 * @param _iRow		row index 
	 * @param _iColumn	column index
	 * @param _fT		output: value of t
	 * @param _fTheta	output: value of theta, always lies within the [0,pi[ interval.
	 */
	virtual void getRayParams(int _iRow, int _iColumn, float32& _fT, float32& _fTheta) const;
	
	/** Returns true if the type of geometry defined in this class is the one specified in _sType.
	 *
	 * @param _sType geometry type to compare to.
	 * @return true if the type of geometry defined in this class is the one specified in _sType. 
	 */
	 virtual bool isOfType(const std::string& _sType) = 0;

	/**
	 * Returns a vector describing the direction of a ray belonging to a certain detector
	 *
	 * @param _iProjectionIndex index of projection
	 * @param _iProjectionIndex index of detector
	 *
	 * @return a unit vector describing the direction
	 */
	 virtual CVector3D getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex) = 0;


	//< For Config unused argument checking
	ConfigCheckData* configCheckData;
	friend class ConfigStackCheck<CProjectionGeometry2D>;
};



//----------------------------------------------------------------------------------------
// Inline member functions
//----------------------------------------------------------------------------------------


inline float32 CProjectionGeometry2D::getExtraDetectorOffset(int _iAngle) const
{
	return m_pfExtraDetectorOffset ? m_pfExtraDetectorOffset[_iAngle] : 0.0f;
}


// Get the initialization state.
inline bool CProjectionGeometry2D::isInitialized() const
{
	return m_bInitialized;
}


// Get the number of detectors.
inline int CProjectionGeometry2D::getDetectorCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iDetectorCount;
}

// Get the width of a single detector (in unit lengths).
inline float32 CProjectionGeometry2D::getDetectorWidth() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fDetectorWidth;
}

// Get the number of projection angles.
inline int CProjectionGeometry2D::getProjectionAngleCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iProjectionAngleCount;
}

// Get pointer to buffer used to store projection angles.
inline const float32* CProjectionGeometry2D::getProjectionAngles() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_pfProjectionAngles;
}

// Get a projection angle, represented in Radians.
inline float32 CProjectionGeometry2D::getProjectionAngle(int _iProjectionIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iProjectionIndex >= 0);
	ASTRA_ASSERT(_iProjectionIndex < m_iProjectionAngleCount);

	return m_pfProjectionAngles[_iProjectionIndex];
}

// Get a projection angle, represented in degrees.
inline float32 CProjectionGeometry2D::getProjectionAngleDegrees(int _iProjectionIndex) const
{
	// basic checks
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iProjectionIndex >= 0);
	ASTRA_ASSERT(_iProjectionIndex < m_iProjectionAngleCount);

	return (m_pfProjectionAngles[_iProjectionIndex] * 180.0f / PI32);
}

// Get T and Theta
inline void CProjectionGeometry2D::getRayParams(int _iRow, int _iColumn, float32& _fT, float32& _fTheta) const
{
	ASTRA_ASSERT(m_bInitialized);
	_fT = indexToDetectorOffset(_iColumn);
	_fTheta = getProjectionAngle(_iRow);
	if (PI <= _fTheta) {
		_fTheta -= PI;
		_fT = -_fT;
	}
}

// detector offset -> detector index
inline int CProjectionGeometry2D::detectorOffsetToIndex(float32 _fOffset) const
{
	int res = (int)(detectorOffsetToIndexFloat(_fOffset) + 0.5f);
	return (res > 0 && res <= m_iDetectorCount) ? res : -1;
}

// detector offset -> detector index (float)
inline float32 CProjectionGeometry2D::detectorOffsetToIndexFloat(float32 _fOffset) const
{
	return (_fOffset / m_fDetectorWidth) + ((m_iDetectorCount-1.0f) * 0.5f);
}

// detector index -> detector offset
inline float32 CProjectionGeometry2D::indexToDetectorOffset(int _iIndex) const
{
	return (_iIndex - (m_iDetectorCount-1.0f) * 0.5f) * m_fDetectorWidth;
}

// sinogram index -> angle and detecor index
inline void CProjectionGeometry2D::indexToAngleDetectorIndex(int _iIndex, int& _iAngleIndex, int& _iDetectorIndex) const
{
	_iAngleIndex = _iIndex / m_iDetectorCount;
	_iDetectorIndex = _iIndex % m_iDetectorCount;
}

} // end namespace astra

#endif /* _INC_ASTRA_PROJECTIONGEOMETRY2D */
