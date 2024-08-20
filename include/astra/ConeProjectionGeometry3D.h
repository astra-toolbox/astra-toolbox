/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
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

#ifndef _INC_ASTRA_CONEPROJECTIONGEOMETRY3D
#define _INC_ASTRA_CONEPROJECTIONGEOMETRY3D

#include "ProjectionGeometry3D.h"

namespace astra
{

/**
 * This class defines a 3D cone beam projection geometry. 
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorRowCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorColCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorSpacingX, float, Width of each detector.}
 * \astra_xml_item{DetectorSpacingY, float, Width of each detector.}
 * \astra_xml_item{ProjectionAngles, vector of float, projection angles in radians.}
 * \astra_xml_item{DistanceOriginDetector, float, Distance between the center of rotation and the detectorarray.}
 * \astra_xml_item{DistanceOriginSource, float, Distance between the center of rotation the the x-ray source.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('cone');\n
 *		proj_geom.DetectorRowCount = 512;\n
 *		proj_geom.DetectorColCount = 512;\n
 *		proj_geom.DetectorSpacingX = 1.0;\n
 *		proj_geom.DetectorSpacingY = 1.0;\n
 *		proj_geom.ProjectionAngles = linspace(0,pi,100);\n
 *		proj_geom.DistanceOriginDetector = 10000;\n
 *		proj_geom.DistanceOriginSource = 10000;\n
 * }
 */
class _AstraExport CConeProjectionGeometry3D : public CProjectionGeometry3D
{
protected:

	/**
	 * Distance from the origin of the coordinate system to the source.
	 */
	float32 m_fOriginSourceDistance;

	/**
	 * Distance from the origin of the coordinate system to the detector (i.e., the distance between the origin and its orthogonal projection
	 * onto the detector array).
	 */
	float32 m_fOriginDetectorDistance;

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling initialize() 
	 * is not allowed, except calling the member function isInitialized().
	 */
	CConeProjectionGeometry3D();

	/** Constructor. Create an instance of the CParallelProjectionGeometry3D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorRowCount Number of rows of detectors.
	 *  @param _iDetectorColCount Number of columns detectors.
	 *  @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _fDetectorHeight Height of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. All angles 
	 *                             are represented in radians and lie in the [0,2pi[ interval.
	 */
	CConeProjectionGeometry3D(int _iProjectionAngleCount,
							  int _iDetectorRowCount,
							  int _iDetectorColCount,
							  float32 _fDetectorWidth,
							  float32 _fDetectorHeight,
							  const float32* _pfProjectionAngles,
							  float32 _fOriginSourceDistance,
							  float32 _fOriginDetectorDistance);

	/** Copy constructor. 
	 */
	CConeProjectionGeometry3D(const CConeProjectionGeometry3D& _projGeom);

	/** Destructor.
	 */
	~CConeProjectionGeometry3D();

	/** Initialize the geometry with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize the geometry. If the object has been initialized before, the object is reinitialized 
	 * and memory is freed and reallocated if necessary.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorRowCount Number of rows of detectors.
	 *  @param _iDetectorColCount Number of columns detectors.
	 *  @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _fDetectorHeight Height of a detector cell, in unit lengths. All detector cells are assumed to have equal height.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. All angles 
	 *                             are represented in radians and lie in the [0,2pi[ interval.
	 */
	bool initialize(int _iProjectionAngleCount,
					int _iDetectorRowCount,
					int _iDetectorColCount,
					float32 _fDetectorWidth,
					float32 _fDetectorHeight,
					const float32* _pfProjectionAngles,
					float32 _fOriginSourceDistance,
					float32 _fOriginDetectorDistance);


	/** Create a hard copy. 
	*/
	virtual CProjectionGeometry3D* clone() const;

	/** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(const CProjectionGeometry3D*) const;

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const;

	/** Returns true if the type of geometry defined in this class is the one specified in _sType.
	 *
	 * @param _sType geometry type to compare to.
	 * @return true if _sType == "cone".
	 */
	virtual bool isOfType(const std::string& _sType) const;

	/** Returns the distance from the origin of the coordinate system to the source.
	 *
	 * @return Distance from the origin of the coordinate system to the source
	 */
	float32 getOriginSourceDistance() const;
	
	/** Returns the distance from the origin of the coordinate system to the detector 
	 * (i.e., the distance between the origin and its orthogonal projection onto the detector array).
	 *
	 * @return Distance from the origin of the coordinate system to the detector
	 */
	float32 getOriginDetectorDistance() const;

	/** Returns the distance from the source to the detector
	 * (i.e., the distance between the source and its orthogonal projection onto the detector array).
	 *
	 * @return Distance from the source to the detector
	 */
	float32 getSourceDetectorDistance() const;

	virtual void projectPoint(double fX, double fY, double fZ,
	                          int iAngleIndex,
	                          double &fU, double &fV) const;

};

// Returns the distance from the origin of the coordinate system to the source.
inline float32 CConeProjectionGeometry3D::getOriginSourceDistance() const
{
	return m_fOriginSourceDistance;
}


// Returns the distance from the origin of the coordinate system to the detector.
inline float32 CConeProjectionGeometry3D::getOriginDetectorDistance() const
{
	return m_fOriginDetectorDistance;
}


// Returns the distance from the source to the detector.
inline float32 CConeProjectionGeometry3D::getSourceDetectorDistance() const
{
	return (m_fOriginSourceDistance + m_fOriginDetectorDistance);
}


} // namespace astra

#endif /* _INC_ASTRA_CONEPROJECTIONGEOMETRY3D */
