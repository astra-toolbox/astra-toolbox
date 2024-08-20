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

#ifndef _INC_ASTRA_PARALLELVECPROJECTIONGEOMETRY3D
#define _INC_ASTRA_PARALLELVECPROJECTIONGEOMETRY3D

#include "ProjectionGeometry3D.h"
#include "GeometryUtil3D.h"

namespace astra
{

/**
 * This class defines a 3D parallel beam projection geometry. 
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorRowCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorColCount, int, Number of detectors for each projection.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('parallel3d_vec');\n
 *		proj_geom.DetectorRowCount = 512;\n
 *		proj_geom.DetectorColCount = 512;\n
 *		proj_geom.Vectors = V;\n
 * }
 *
 * \par Vectors
 * Vectors is a matrix containing the actual geometry. Each row corresponds
 * to a single projection, and consists of:
 * ( rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
 *      ray: the ray direction
 *      d  : the centre of the detector plane
 *      u  : the vector from detector pixel (0,0) to (0,1)
 *      v  : the vector from detector pixel (0,0) to (1,0)
 */
class _AstraExport CParallelVecProjectionGeometry3D : public CProjectionGeometry3D
{
protected:

	SPar3DProjection *m_pProjectionAngles;

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling initialize() 
	 * is not allowed, except calling the member function isInitialized().
	 */
	CParallelVecProjectionGeometry3D();

	/** Constructor. Create an instance of the CParallelProjectionGeometry3D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorRowCount Number of rows of detectors.
	 *  @param _iDetectorColCount Number of columns detectors.
	 *  @param _pProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 */
	CParallelVecProjectionGeometry3D(int _iProjectionAngleCount, 
	                                 int _iDetectorRowCount, 
	                                 int _iDetectorColCount,
	                                 const SPar3DProjection* _pProjectionAngles);

	/** Copy constructor. 
	 */
	CParallelVecProjectionGeometry3D(const CParallelVecProjectionGeometry3D& _projGeom);

	/** Destructor.
	 */
	~CParallelVecProjectionGeometry3D();

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
	 *  @param _pProjectionAngles Pointer to an array of projection angles. The angles will be copied from this array.
	 */
	bool initialize(int _iProjectionAngleCount, 
					int _iDetectorRowCount, 
					int _iDetectorColCount,
					const SPar3DProjection* _pProjectionAngles);


	virtual bool _check();

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
	 * @return true if _sType == "parallel3d_vec".
	 */
	 virtual bool isOfType(const std::string& _sType) const;

	const SPar3DProjection* getProjectionVectors() const { return m_pProjectionAngles; }

	virtual void projectPoint(double fX, double fY, double fZ,
	                          int iAngleIndex,
	                          double &fU, double &fV) const;

protected:
	virtual bool initializeAngles(const Config& _cfg);
};

} // namespace astra

#endif /* _INC_ASTRA_PARALLELVECPROJECTIONGEOMETRY3D */
