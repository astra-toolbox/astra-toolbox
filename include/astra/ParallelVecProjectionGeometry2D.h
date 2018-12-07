/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#ifndef _INC_ASTRA_PARALLELVECPROJECTIONGEOMETRY2D
#define _INC_ASTRA_PARALLELVECPROJECTIONGEOMETRY2D

#include "ProjectionGeometry2D.h"
#include "GeometryUtil2D.h"

namespace astra
{

/**
 * This class defines a 2D parallel beam projection geometry. 
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorCount, int, Number of detectors for each projection.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('parallel_vec');\n
 *		proj_geom.DetectorCount = 512;\n
 *		proj_geom.Vectors = V;\n
 * }
 *
 * \par Vectors
 * Vectors is a matrix containing the actual geometry. Each row corresponds
 * to a single projection, and consists of:
 * ( rayX, rayY, dX, dY, uX, uY)
 *      ray: the ray direction
 *      d  : the centre of the detector line
 *      u  : the vector from detector pixel (0) to (1)
 */
class _AstraExport CParallelVecProjectionGeometry2D : public CProjectionGeometry2D
{
protected:

	SParProjection *m_pProjectionAngles;

public:

	/** Default constructor. Sets all variables to zero. Note that this constructor leaves the object in an unusable state and must
	 * be followed by a call to init(). 
	 */
	CParallelVecProjectionGeometry2D();

	/** Constructor.
	 *
	 * @param _iProjectionAngleCount Number of projection angles.
	 * @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 * @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array. 
	 */
	CParallelVecProjectionGeometry2D(int _iProjectionAngleCount, 
	                                int _iDetectorCount, 
	                                const SParProjection* _pfProjectionAngles);

	/** Copy constructor. 
	 */
	CParallelVecProjectionGeometry2D(const CParallelVecProjectionGeometry2D& _projGeom);

	/** Assignment operator.
	 */
	CParallelVecProjectionGeometry2D& operator=(const CParallelVecProjectionGeometry2D& _other);

	/** Destructor.
	 */
	virtual ~CParallelVecProjectionGeometry2D();

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
	 * @param _pfProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 */
	bool initialize(int _iProjectionAngleCount, 
	                int _iDetectorCount, 
	                const SParProjection* _pfProjectionAngles);

	virtual bool _check();

	/** Create a hard copy. 
	*/
	virtual CProjectionGeometry2D* clone();

	/** Returns true if the type of geometry defined in this class is the one specified in _sType.
	 *
	 * @param _sType geometry type to compare to.
	 * @return true if _sType == "fanflat_vec".
	 */
	 virtual bool isOfType(const std::string& _sType);

    /** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(CProjectionGeometry2D*) const;

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const;


	/** Get the value for t and theta, based upon the row and column index.
	 *
	 * @param _iRow		row index 
	 * @param _iColumn	column index
	 * @param _fT		output: value of t
	 * @param _fTheta	output: value of theta, always lies within the [0,pi[ interval.
	 */
	virtual void getRayParams(int _iRow, int _iColumn, float32& _fT, float32& _fTheta) const;

	/**
	 * Returns a vector describing the direction of a ray belonging to a certain detector
	 *
	 * @param _iProjectionIndex index of projection
	 * @param _iProjectionIndex index of detector
	 *
	 * @return a unit vector describing the direction
	 */
	virtual CVector3D getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex);

	const SParProjection* getProjectionVectors() const { return m_pProjectionAngles; }

protected:
	virtual bool initializeAngles(const Config& _cfg);
};

} // namespace astra

#endif /* _INC_ASTRA_PARALLELVECPROJECTIONGEOMETRY2D */
