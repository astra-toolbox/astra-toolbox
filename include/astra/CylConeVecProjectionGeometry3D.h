/*
-----------------------------------------------------------------------
Copyright: 2021, CWI, Amsterdam
           2021, University of Cambridge

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------------------------------------------------------
*/

#ifndef _INC_ASTRA_CYLCONEVECPROJECTIONGEOMETRY3D
#define _INC_ASTRA_CYLCONEVECPROJECTIONGEOMETRY3D

#include "ProjectionGeometry3D.h"
#include "GeometryUtil3D.h"

namespace astra
{

/**
 * This class defines a 3D cone beam projection geometry for a detector curved in one direction. 
 *
 * \par XML Configuration
 * \astra_xml_item{DetectorRowCount, int, Number of detectors for each projection.}
 * \astra_xml_item{DetectorColCount, int, Number of detectors for each projection.}
 * \astra_xml_item{Vectors, matrix defining the 3D position of source and detector.}
 *
 * \par MATLAB example
 * \astra_code{
 *		proj_geom = astra_struct('cyl_cone_vec');\n
 *		proj_geom.DetectorRowCount = 512;\n
 *		proj_geom.DetectorColCount = 512;\n
 *		proj_geom.Vectors = V;\n
 * }
 *
 * \par Vectors
 * Vectors is a matrix containing the actual geometry. Each row corresponds
 * to a single projection, and consists of:
 * ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ, R )
 *      src: the ray source
 *      c  : the centre of the detector
 *      u  : tangent from det pixel (0,0) to (0,1); length is the size of
 *           a detector pixel along the cylinder
 *      v  : the vector from detector pixel (0,0) to (1,0)
 *      R  : radius of the cylindrical detector
 *
 *  NB: The meaning of d and u is different than for cone_vec
 */
class _AstraExport CCylConeVecProjectionGeometry3D : public CProjectionGeometry3D
{
protected:

	SCylConeProjection *m_pProjectionAngles;

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling initialize() 
	 * is not allowed, except calling the member function isInitialized().
	 */
	CCylConeVecProjectionGeometry3D();

	/** Constructor. Create an instance of the CConeVecProjectionGeometry3D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorRowCount Number of rows of detectors.
	 *  @param _iDetectorColCount Number of columns detectors.
	 *  @param _pProjectionAngles Pointer to an array of projection vectors. The data will be copied from this array.
	 *  @param _fCylRadius Number of rows of detectors.
	 */
	CCylConeVecProjectionGeometry3D(int _iProjectionAngleCount, 
	                             int _iDetectorRowCount, 
	                             int _iDetectorColCount,
	                             const SCylConeProjection* _pProjectionAngles);

	/** Copy constructor. 
	 */
	CCylConeVecProjectionGeometry3D(const CCylConeVecProjectionGeometry3D& _projGeom);

	/** Destructor.
	 */
	~CCylConeVecProjectionGeometry3D();

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
	 *  @param _pProjectionAngles Pointer to an array of projection vectors. The data will be copied from this array.
	 */
	bool initialize(int _iProjectionAngleCount, 
					int _iDetectorRowCount, 
					int _iDetectorColCount,
	                const SCylConeProjection* _pProjectionAngles);

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
	 * @return true if _sType == "cyl_cone_vec".
	 */
	 virtual bool isOfType(const std::string& _sType) const;

	const SCylConeProjection* getProjectionVectors() const { return m_pProjectionAngles; }

	virtual void getProjectedBBoxSingleAngle(int iAngle,
	                              double fXMin, double fXMax,
	                              double fYMin, double fYMax,
	                              double fZMin, double fZMax,
	                              double &fUMin, double &fUMax,
	                              double &fVMin, double &fVMax) const;

	virtual void projectPoint(double fX, double fY, double fZ,
	                          int iAngleIndex,
	                          double &fU, double &fV) const;

};

} // namespace astra

#endif /* _INC_ASTRA_CYLCONEVECPROJECTIONGEOMETRY3D */
