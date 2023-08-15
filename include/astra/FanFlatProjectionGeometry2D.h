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

#ifndef _INC_ASTRA_FANFLATPROJECTIONGEOMETRY2D
#define _INC_ASTRA_FANFLATPROJECTIONGEOMETRY2D

#include "ProjectionGeometry2D.h"
#include "FanFlatVecProjectionGeometry2D.h"

#include <cmath>

namespace astra
{

/**
 * This class defines a 2D fan beam geometry with a flat detector that has equally spaced detector cells. 
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorWidth, float, Width of each detector.}
 * \astra_xml_item{ProjectionAngles, vector of float, projection angles in radians.}
 * \astra_xml_item{DistanceOriginDetector, float, Distance between the center of rotation and the detectorarray.}
 * \astra_xml_item{DistanceOriginSource, float, Distance between the center of rotation the the x-ray source.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('fanflat');\n
 *		proj_geom.DetectorCount = 512;\n
 *		proj_geom.DetectorWidth = 1.0;\n
 *		proj_geom.ProjectionAngles = linspace(0,pi,100);\n
 *		proj_geom.DistanceOriginDetector = 300;\n
 *		proj_geom.DistanceOriginSource = 300;\n
 * }
 */
class _AstraExport CFanFlatProjectionGeometry2D : public CProjectionGeometry2D
{
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

	/** Default constructor. Sets all variables to zero. Note that this constructor leaves the object in an unusable state and must
	 * be followed by a call to init(). 
	 */
	CFanFlatProjectionGeometry2D();

	/** Constructor.
	 *
	 * @param _iProjectionAngleCount Number of projection angles.
	 * @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 * @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 * @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. 
	 *		  All angles are represented in radians.
	 * @param _fOriginSourceDistance Distance from the origin of the coordinate system to the source
	 * @param _fOriginDetectorDistance Distance from the origin of the coordinate system to the detector
	 */
	CFanFlatProjectionGeometry2D(int _iProjectionAngleCount, 
								 int _iDetectorCount, 
								 float32 _fDetectorWidth, 
								 const float32* _pfProjectionAngles,
								 float32 _fOriginSourceDistance, 
								 float32 _fOriginDetectorDistance);

	/** Copy constructor. 
	 */
	CFanFlatProjectionGeometry2D(const CFanFlatProjectionGeometry2D& _projGeom);

	/** Assignment operator.
	 */
	CFanFlatProjectionGeometry2D& operator=(const CFanFlatProjectionGeometry2D& _other);

	/** Destructor.
	 */
	virtual ~CFanFlatProjectionGeometry2D();

	/** Initialize the geometry with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialization. This function MUST be called after using the default constructor and MAY be called to 
	 * reset a previously initialized object.
	 *
	 * @param _iProjectionAngleCount Number of projection angles.
	 * @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 * @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 * @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 * @param _fOriginSourceDistance Distance from the origin of the coordinate system to the source
	 * @param _fOriginDetectorDistance Distance from the origin of the coordinate system to the detector
	 */
	bool initialize(int _iProjectionAngleCount, 
					int _iDetectorCount, 
					float32 _fDetectorWidth, 
					const float32* _pfProjectionAngles,
					float32 _fOriginSourceDistance, 
					float32 _fOriginDetectorDistance);

	/** Create a hard copy. 
	*/
	virtual CProjectionGeometry2D* clone();

	/** Returns true if the type of geometry defined in this class is the one specified in _sType.
	 *
	 * @param _sType geometry type to compare to.
	 * @return true if _sType == "fanflat".
	 */
	 virtual bool isOfType(const std::string& _sType);

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const;

    /** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(CProjectionGeometry2D*) const;

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

	/** Create a vector geom
	*/
	CFanFlatVecProjectionGeometry2D* toVectorGeometry();	
};



// Returns the distance from the origin of the coordinate system to the source.
inline float32 CFanFlatProjectionGeometry2D::getOriginSourceDistance() const
{
	return m_fOriginSourceDistance;
}


// Returns the distance from the origin of the coordinate system to the detector.
inline float32 CFanFlatProjectionGeometry2D::getOriginDetectorDistance() const
{
	return m_fOriginDetectorDistance;
}


// Returns the distance from the source to the detector.
inline float32 CFanFlatProjectionGeometry2D::getSourceDetectorDistance() const
{
	return (m_fOriginSourceDistance + m_fOriginDetectorDistance);
}


} // namespace astra

#endif /* _INC_ASTRA_FANFLATPROJECTIONGEOMETRY2D */

