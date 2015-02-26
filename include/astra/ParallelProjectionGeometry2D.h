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

#ifndef _INC_ASTRA_PARALLELPROJECTIONGEOMETRY2D
#define _INC_ASTRA_PARALLELPROJECTIONGEOMETRY2D

#include "ProjectionGeometry2D.h"

namespace astra
{

/**
 * This class defines a 2D parallel beam projection geometry. 
 *
 * <img src="../images/par_proj_geom.svg" width="600"/>
 * This geometry is defined by a number of parameters:
 * - The number of detectors (DetCount).
 *   The distance between the first detector and the projection of the origin \f$O\f$ is equal to 
 *   the distance between the last detector and the projection of \f$O\f$. 
 * - The width of each detector (detector width).  All detectors are equidistant.
 * - A list of projection angles (\f$\theta\f$), measured w.r.t. the y-axis of the volume.  In Radians.  Should lie in the interval \f$[-\frac{\pi}{4},\frac{7\pi}{4}]\f$.
 *
 * This class provides functionality to convert between detector index and detector offset \f$t\f$.
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorWidth, float, Width of each detector.}
 * \astra_xml_item{ProjectionAngles, vector of float, projection angles w.r.t. the y-axis of the volume in radians.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('parallel');\n
 *		proj_geom.DetectorCount = 512;\n
 *		proj_geom.DetectorWidth = 1.0;\n
 *		proj_geom.ProjectionAngles = linspace(0\,pi\,100);\n
 * }
 */
class _AstraExport CParallelProjectionGeometry2D : public CProjectionGeometry2D
{
public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CParallelProjectionGeometry2D();

	/** Constructor. Create an instance of the CParallelProjectionGeometry2D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 *  @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 */
	CParallelProjectionGeometry2D(int _iProjectionAngleCount, 
								  int _iDetectorCount, 
								  float32 _fDetectorWidth, 
								  const float32* _pfProjectionAngles,
								  const float32* _pfExtraDetectorOffsets = 0);

	/** Copy constructor. 
	 */
	CParallelProjectionGeometry2D(const CParallelProjectionGeometry2D& _projGeom);

	/** Destructor.
	 */
	~CParallelProjectionGeometry2D();

	/** Assignment operator.
	 */
	CParallelProjectionGeometry2D& operator=(const CParallelProjectionGeometry2D& _other);

	/** Initialize the geometry with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialization. Initializes an instance of the CProjectionGeometry2D class. If the object has been 
	 * initialized before, the object is reinitialized and memory is freed and reallocated if necessary.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 *  @param _fDetectorWidth Width of a detector cell, in unit lengths. All detector cells are assumed to have equal width.
	 *  @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 */
	bool initialize(int _iProjectionAngleCount, 
					int _iDetectorCount, 
					float32 _fDetectorWidth, 
					const float32* _pfProjectionAngles,
					const float32* _pfExtraDetectorOffsets = 0);

	/** Create a hard copy. 
	*/
	virtual CProjectionGeometry2D* clone();

    /** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(CProjectionGeometry2D*) const;

	/** Returns true if the type of geometry defined in this class is the one specified in _sType.
	 *
	 * @param _sType geometry type to compare to.
	 * @return true if _sType == "parallel".
	 */
	 virtual bool isOfType(const std::string& _sType);

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const;

 	/**
	 * Returns a vector describing the direction of a ray belonging to a certain detector,
	 *   the direction is the same for all detectors in one projection 
	 *
	 * @param _iProjectionIndex index of projection
	 * @param _iProjectionIndex index of detector
	 *
	 * @return a unit vector describing the direction
	 */
	 virtual CVector3D getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex = 0);
};

} // namespace astra

#endif /* _INC_ASTRA_PARALLELPROJECTIONGEOMETRY2D */
