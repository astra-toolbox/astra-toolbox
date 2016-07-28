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

#ifndef _INC_ASTRA_PARALLELPROJECTIONGEOMETRY3D
#define _INC_ASTRA_PARALLELPROJECTIONGEOMETRY3D

#include "ProjectionGeometry3D.h"
#include "ParallelProjectionGeometry2D.h"

namespace astra
{

/**
 * This class defines a 3D parallel beam projection geometry. 
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorRowCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorColCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorSpacingX, float, Width of each detector.}
 * \astra_xml_item{DetectorSpacingY, float, Width of each detector.}
 * \astra_xml_item{ProjectionAngles, vector of float, projection angles in radians.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('parallel');\n
 *		proj_geom.DetectorRowCount = 512;\n
 *		proj_geom.DetectorColCount = 512;\n
 *		proj_geom.DetectorSpacingX = 1.0;\n
 *		proj_geom.DetectorSpacingY = 1.0;\n
 *		proj_geom.ProjectionAngles = linspace(0,pi,100);\n
 * }
 */
class _AstraExport CParallelProjectionGeometry3D : public CProjectionGeometry3D
{
protected:

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling initialize() 
	 * is not allowed, except calling the member function isInitialized().
	 */
	CParallelProjectionGeometry3D();

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
	CParallelProjectionGeometry3D(int _iProjectionAngleCount, 
								  int _iDetectorRowCount, 
								  int _iDetectorColCount,
								  float32 _fDetectorWidth, 
								  float32 _fDetectorHeight, 
								  const float32* _pfProjectionAngles);

	/** Copy constructor. 
	 */
	CParallelProjectionGeometry3D(const CParallelProjectionGeometry3D& _projGeom);

	/** Destructor.
	 */
	~CParallelProjectionGeometry3D();

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
					const float32* _pfProjectionAngles);

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
	 * @return true if _sType == "parallel".
	 */
	 virtual bool isOfType(const std::string& _sType) const;

	 /**
	  * Returns a vector giving the projection direction for a projection and detector index
	  */
	virtual CVector3D getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex) const;

	virtual void projectPoint(double fX, double fY, double fZ,
	                          int iAngleIndex,
	                          double &fU, double &fV) const;
	virtual void backprojectPointX(int iAngleIndex, double fU, double fV,
	                               double fX, double &fY, double &fZ) const;
	virtual void backprojectPointY(int iAngleIndex, double fU, double fV,
	                               double fY, double &fX, double &fZ) const;
	virtual void backprojectPointZ(int iAngleIndex, double fU, double fV,
	                               double fZ, double &fX, double &fY) const;


	 /**
	  * Creates (= allocates) a 2D projection geometry used when projecting one slice using a 2D projector
	  *
	  * @return the 2D geometry, this pointer needs to be delete-ed after use.
	  */
	CParallelProjectionGeometry2D * createProjectionGeometry2D() const;

};

} // namespace astra

#endif /* _INC_ASTRA_PARALLELPROJECTIONGEOMETRY3D */
