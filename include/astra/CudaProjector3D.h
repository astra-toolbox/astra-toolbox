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

#ifndef INC_ASTRA_CUDAPROJECTOR3D
#define INC_ASTRA_CUDAPROJECTOR3D

#ifdef ASTRA_CUDA

#include <cmath>
#include <vector>

#include "Globals.h"
#include "Config.h"
#include "Projector3D.h"
#include "../../cuda/3d/astra3d.h"

namespace astra
{

/** This is a three-dimensional CUDA-projector.
 *  It is essentially a fake projector, containing settings relevant for the
 *  actual CUDA code.
 */
class _AstraExport CCudaProjector3D : public CProjector3D
{

protected:

	/** Check variable values.
	 */
	bool _check();

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 * Should only be used by constructors.  Otherwise use the clear() function.
	 */
	void _clear();

public:

	// type of the projector, needed to register with CProjectorFactory
	static std::string type;

	/**
	 * Default Constructor.
	 */
	CCudaProjector3D();

	/** Destructor, is virtual to show that we are aware subclass destructor is called.
	 */
	virtual ~CCudaProjector3D();
	
	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 */
	void clear();

	/** Initialize the projector with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	virtual void computeSingleRayWeights(int _iProjectionIndex, 
										 int _iSliceIndex,
										 int _iDetectorIndex, 
										 SPixelWeight* _pWeightedPixels,
		                                 int _iMaxPixelCount, 
										 int& _iStoredPixelCount) {}
	virtual int getProjectionWeightsCount(int _iProjectionIndex) { return 0; }
	template <typename Policy>
	void project(Policy& _policy) {}
	template <typename Policy>
	void projectSingleProjection(int _iProjection, Policy& _policy) {}
	template <typename Policy>
	void projectSingleRay(int _iProjection, int _iSlice, int _iDetector, Policy& _policy) {}



	/** Return the  type of this projector.
	 *
	 * @return identification type of this projector
	 */
	virtual std::string getType() { return type; }

	/** get a description of the class
	 *
	 * @return description string
	 */
	virtual std::string description() const;


	Cuda3DProjectionKernel getProjectionKernel() const { return m_projectionKernel; }
	int getVoxelSuperSampling() const { return m_iVoxelSuperSampling; }
	int getDetectorSuperSampling() const { return m_iDetectorSuperSampling; }
	int getGPUIndex() const { return m_iGPUIndex; }

protected:

	Cuda3DProjectionKernel m_projectionKernel;
	int m_iVoxelSuperSampling;
	int m_iDetectorSuperSampling;
	int m_iGPUIndex;

};


} // namespace astra

#endif // ASTRA_CUDA

#endif /* INC_ASTRA_CUDAPROJECTOR3D */
