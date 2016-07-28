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

#ifndef _INC_ASTRA_CUDAPROJECTOR2D
#define _INC_ASTRA_CUDAPROJECTOR2D

#include "ParallelProjectionGeometry2D.h"
#include "Float32Data2D.h"
#include "Projector2D.h"
#include "../../cuda/2d/astra.h"

namespace astra
{


/** This is a two-dimensional CUDA-projector.
 *  It is essentially a fake projector, containing settings relevant for the
 *  actual CUDA code.
 */
class _AstraExport CCudaProjector2D : public CProjector2D {

protected:
	
	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - no NULL pointers
	 * - all sub-objects are initialized properly
	 * - blobvalues are ok
	 */
	virtual bool _check();

public:

	// type of the projector, needed to register with CProjectorFactory
	static std::string type;

	/** Default constructor.
	 */
	CCudaProjector2D();

	/** Constructor.
	 * 
	 * @param _pProjectionGeometry		Information class about the geometry of the projection.  Will be HARDCOPIED.
	 * @param _pReconstructionGeometry	Information class about the geometry of the reconstruction volume. Will be HARDCOPIED.
	 */
	CCudaProjector2D(CParallelProjectionGeometry2D* _pProjectionGeometry, 
	                 CVolumeGeometry2D* _pReconstructionGeometry);
	
	~CCudaProjector2D();

	/** Initialize the projector with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Clear this class.
	 */
	virtual void clear();


	virtual int getProjectionWeightsCount(int _iProjectionIndex) { return 0; }

	virtual void computeSingleRayWeights(int _iProjectionIndex, 
										 int _iDetectorIndex, 
										 SPixelWeight* _pWeightedPixels,
		                                 int _iMaxPixelCount, 
										 int& _iStoredPixelCount) {}

	virtual std::vector<SDetector2D> projectPoint(int _iRow, int _iCol)
		{ std::vector<SDetector2D> x; return x; }

	template <typename Policy>
	void project(Policy& _policy) {}
	template <typename Policy>
	void projectSingleProjection(int _iProjection, Policy& _policy) {}
	template <typename Policy>
	void projectSingleRay(int _iProjection, int _iDetector, Policy& _policy) {}


	/** Return the  type of this projector.
	 *
	 * @return identification type of this projector
	 */
	virtual std::string getType();

	/** get a description of the class
	 *
	 * @return description string
	 */
	virtual std::string description() const;


	Cuda2DProjectionKernel getProjectionKernel() const { return m_projectionKernel; }
	int getVoxelSuperSampling() const { return m_iVoxelSuperSampling; }
	int getDetectorSuperSampling() const { return m_iDetectorSuperSampling; }
	int getGPUIndex() const { return m_iGPUIndex; }

protected:

	Cuda2DProjectionKernel m_projectionKernel;
	int m_iVoxelSuperSampling;
	int m_iDetectorSuperSampling;
	int m_iGPUIndex;
};

//----------------------------------------------------------------------------------------

inline std::string CCudaProjector2D::getType() 
{ 
	return type; 
}

} // namespace astra

#endif 

