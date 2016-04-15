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

#ifndef INC_ASTRA_PROJECTOR2D
#define INC_ASTRA_PROJECTOR2D

#include <cmath>
#include <vector>

#include "Globals.h"
#include "Config.h"
#include "Float32Data2D.h"
#include "ParallelProjectionGeometry2D.h"
#include "ProjectionGeometry2D.h"
#include "VolumeGeometry2D.h"

namespace astra
{

class CSparseMatrix;


/** This is a base interface class for a two-dimensional projector.  Each subclass should at least 
 * implement the core projection functions computeProjectionRayWeights and projectPoint.   For 
 * extra efficiency one might also like to overwrite other functions such as computeProjectionRayWeights,
 * computeRayForwardProj_ART, ...
 *
 * \par XML Configuration
 * \astra_xml_item{ProjectionGeometry, xml node, The geometry of the projection.}
 * \astra_xml_item{VolumeGeometry, xml node, The geometry of the volume.}
 */
class _AstraExport CProjector2D
{

protected:
	CProjectionGeometry2D* m_pProjectionGeometry; ///< Used projection geometry
	CVolumeGeometry2D* m_pVolumeGeometry; ///< Used volume geometry
	bool m_bIsInitialized; ///< Has this class been initialized?

	/** Default Constructor.
	 */
	CProjector2D();

	/** Constructor.
	 * 
	 * @param _pProjectionGeometry		Information class about the geometry of the projection. Will be HARDCOPIED.
	 * @param _pVolumeGeometry	Information class about the geometry of the reconstruction volume. Will be HARDCOPIED.
	 */
	CProjector2D(CProjectionGeometry2D* _pProjectionGeometry, CVolumeGeometry2D* _pVolumeGeometry);

	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - no NULL pointers
	 * - all sub-objects are initialized properly
	 */
	virtual bool _check();

public:
	
	/** Destructor.
	 */
	virtual ~CProjector2D();

	/** Clear this class.
	 */
	virtual void clear();

	/** Initialize the projector with a config object.
	 * This function does not set m_bInitialized to true.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Fetch the Projection Geometry of this projector.
	 *
	 * @return Projection Geometry class.
	 */
	CProjectionGeometry2D* getProjectionGeometry();

	/** Fetch the Volume Geometry of this projector.
	 *
	 * @return Volume Geometry class.
	 */
	CVolumeGeometry2D* getVolumeGeometry();

	/** Compute the pixel weights for a single ray, from the source to a detector pixel. 
	 *
	 * @param _iProjectionIndex	Index of the projection 
	 * @param _iDetectorIndex	Index of the detector pixel
	 * @param _pWeightedPixels	Pointer to a pre-allocated array, consisting of _iMaxPixelCount elements
	 *							of type SPixelWeight. On return, this array contains a list of the index
	 *							and weight for all pixels on the ray.
	 * @param _iMaxPixelCount	Maximum number of pixels (and corresponding weights) that can be stored in _pWeightedPixels.
	 *							This number MUST be greater than the total number of pixels on the ray.
	 * @param _iStoredPixelCount On return, this variable contains the total number of pixels on the 
	 *                           ray (that have been stored in the list _pWeightedPixels). 
     */
	virtual void computeSingleRayWeights(int _iProjectionIndex, 
										 int _iDetectorIndex, 
										 SPixelWeight *_pWeightedPixels,
		                                 int _iMaxPixelCount, 
										 int &_iStoredPixelCount) = 0;

	/** Compute the pixel weights for all rays in a single projection, from the source to a each of the 
	 * detector pixels. All pixels and their weights are stored consecutively in the array _pWeightedPixels. 
	 * The array starts with all pixels on the first ray, followed by all pixels on the second ray, the third 
	 * ray, etc. Note that a pixel may occur in the list more than once, as it can be on several rays. 
	 *
	 * @param _iProjectionIndex Index of the projection (zero-based).
	 * @param _pfWeightedPixels	Pointer to a pre-allocated array, consisting of getProjectionWeightsCount() 
	 *							elements of type SPixelWeight. On return, this array contains a list of 
	 *							the index and weight for all pixels on each of the rays. The elements for 
	 *							every ray start at equal offsets (ray_index * _pWeightedPixels / ray_count).
	 * @param _piRayStoredPixelCount Pointer to a pre-allocated array, containing a single integer for each 
	 *								 ray in the projection. On return, this array contains the number of 
	 *								 pixels on the ray, for each ray in the given projection. 
     */
	virtual void computeProjectionRayWeights(int _iProjectionIndex, 
											 SPixelWeight* _pfWeightedPixels, 
											 int* _piRayStoredPixelCount);

	/** Create a list of detectors that are influenced by point [_iRow, _iCol].
	 *
	 * @param _iRow row of the point
	 * @param _iCol column of the point
	 * @return list of SDetector2D structs
	 */
	virtual std::vector<SDetector2D> projectPoint(int _iRow, int _iCol) = 0;

	/** Returns the number of weights required for storage of all weights of one projection ray.
	 *
	 * @param _iProjectionIndex Index of the projection (zero-based).
	 * @return Size of buffer (given in SPixelWeight elements) needed to store weighted pixels.
	 */
	virtual int getProjectionWeightsCount(int _iProjectionIndex) = 0;

	/** Returns the projection as an explicit sparse matrix.
	 * @return a newly allocated CSparseMatrix. Delete afterwards.
	 */
	CSparseMatrix* getMatrix();

	/** Has the projector been initialized?
	 *
	 * @return initialized successfully
	 */
	bool isInitialized() const;

	/** get a description of the class
	 *
	 * @return description string
	 */
	virtual std::string description() const {return " ";};

	virtual std::string getType() { return " "; }

private:
	//< For Config unused argument checking
	ConfigCheckData* configCheckData;
	friend class ConfigStackCheck<CProjector2D>;
};

// inline functions
inline bool CProjector2D::isInitialized() const { return m_bIsInitialized; }
inline CProjectionGeometry2D* CProjector2D::getProjectionGeometry() { return m_pProjectionGeometry; }
inline CVolumeGeometry2D* CProjector2D::getVolumeGeometry() { return m_pVolumeGeometry; }





} // namespace astra

#endif /* INC_ASTRA_PROJECTOR2D */
