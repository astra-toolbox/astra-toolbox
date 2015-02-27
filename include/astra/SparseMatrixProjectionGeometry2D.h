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

#ifndef _INC_ASTRA_SPARSEMATRIXPROJECTIONGEOMETRY2D
#define _INC_ASTRA_SPARSEMATRIXPROJECTIONGEOMETRY2D

#include "ProjectionGeometry2D.h"

namespace astra
{

class CSparseMatrix;

/**
 * This class defines a projection geometry determined by an arbitrary
 * sparse matrix.
 *
 * The projection data is assumed to be grouped by 'angle' and 'detector pixel'.
 * This does not have any effect on the algorithms, but only on the
 * way the projection data is stored and accessed.
 */
class _AstraExport CSparseMatrixProjectionGeometry2D : public CProjectionGeometry2D
{
public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CSparseMatrixProjectionGeometry2D();

	/** Constructor. Create an instance of the CSparseMatrixProjectionGeometry2D class.
	 *
	 *  @param _iProjectionAngleCount Number of projection angles.
	 *  @param _iDetectorCount Number of detectors, i.e., the number of detector measurements for each projection angle.
	 *  @param _pMatrix Pointer to a CSparseMatrix. The caller is responsible for keeping this matrix valid until it is no longer required.
	 */
	CSparseMatrixProjectionGeometry2D(int _iProjectionAngleCount, 
								  int _iDetectorCount, 
								  const CSparseMatrix* _pMatrix);

	/** Copy constructor. 
	 */
	CSparseMatrixProjectionGeometry2D(const CSparseMatrixProjectionGeometry2D& _projGeom);

	/** Destructor.
	 */
	~CSparseMatrixProjectionGeometry2D();

	/** Assignment operator.
	 */
	CSparseMatrixProjectionGeometry2D& operator=(const CSparseMatrixProjectionGeometry2D& _other);

	/** Initialize the geometry with a config object. This does not allow
	 *  setting a matrix. Use the setMatrix() method for that afterwards.
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
	 *  @param _pMatrix Pointer to a CSparseMatrix. The caller is responsible for keeping this matrix valid until it is no longer required.
	 */
	bool initialize(int _iProjectionAngleCount, 
					int _iDetectorCount, 
					const CSparseMatrix* _pMatrix);

	/** Set the associated sparse matrix. The previous one is deleted.
	 *
	 * @param _pMatrix Pointer to a CSparseMatrix. The caller is responsible for keeping this matrix valid until it is no longer required.
	 * @return initialization successful?
	 */
	bool setMatrix(CSparseMatrix* _pMatrix);

	/** Get a pointer to the associated sparse matrix.
	 * @return the associated sparse matrix
	 */
	const CSparseMatrix* getMatrix() const { return m_pMatrix; }

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
	 * Returns a vector describing the direction of a ray belonging to a certain detector
	 *
	 * @param _iProjectionIndex index of projection
	 * @param _iProjectionIndex index of detector
	 *
	 * @return a unit vector describing the direction
	 */
	 virtual CVector3D getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex);

protected:

	/** Check this object.
	 *
	 * @return object initialized
	 */
	bool _check();

	const CSparseMatrix* m_pMatrix;
};

} // namespace astra

#endif /* _INC_ASTRA_SPARSEMATRIXPROJECTIONGEOMETRY2D */
