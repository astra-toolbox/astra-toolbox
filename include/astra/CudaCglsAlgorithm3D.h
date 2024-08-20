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

#ifndef _INC_ASTRA_CUDACGLSALGORITHM3D
#define _INC_ASTRA_CUDACGLSALGORITHM3D

#include "Globals.h"
#include "Config.h"

#include "Algorithm.h"

#include "Data3D.h"
#include "ReconstructionAlgorithm3D.h"

#ifdef ASTRA_CUDA

namespace astra {

class AstraCGLS3d;

/**
 * \brief
 * This class contains the 3D implementation of the CGLS algorithm
 *
 */
class _AstraExport CCudaCglsAlgorithm3D : public CReconstructionAlgorithm3D {

protected:

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - no NULL pointers
	 * - all sub-objects are initialized properly
	 * - the projector is compatible with both data objects
	 */
	virtual bool _check();

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, does not initialize the object.
	 */
	CCudaCglsAlgorithm3D();

	/** Constructor with initialization.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pProjectionData	ProjectionData3D object containing the projection data.
	 * @param _pReconstruction	VolumeData3D object for storing the reconstructed volume.
	 */
	CCudaCglsAlgorithm3D(CProjector3D* _pProjector,
	                     CFloat32ProjectionData3D* _pProjectionData,
	                     CFloat32VolumeData3D* _pReconstruction);
	
	/** Copy constructor.
	 */
	CCudaCglsAlgorithm3D(const CCudaCglsAlgorithm3D&);

	/** Destructor.
	 */
	virtual ~CCudaCglsAlgorithm3D();

	/** Clear this class.
	 */
/*	virtual void clear();*/

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pProjectionData	ProjectionData3D object containing the projection data.
	 * @param _pReconstruction	VolumeData3D object for storing the reconstructed volume.
	 * @return initialization successful?
	 */
	bool initialize(CProjector3D* _pProjector,
					CFloat32ProjectionData3D* _pSinogram,
					CFloat32VolumeData3D* _pReconstruction);

	/** Perform a number of iterations.
	 *
	 * @param _iNrIterations amount of iterations to perform.
	 */
	virtual void run(int _iNrIterations = 0);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

	/**  
	 * Sets the index of the used GPU index: first GPU has index 0
	 *
	 * @param _iGPUIndex New GPU index.
	 */
	void setGPUIndex(int _iGPUIndex) { m_iGPUIndex = _iGPUIndex; }

	/** Get the norm of the residual image.
	 *  Only a few algorithms support this method.
	 *
	 * @param _fNorm if supported, the norm is returned here
	 * @return true if this operation is supported
	 */
	virtual bool getResidualNorm(float32& _fNorm);

protected:

	AstraCGLS3d* m_pCgls;

	int m_iGPUIndex;
	bool m_bAstraCGLSInit;
	int m_iDetectorSuperSampling;
	int m_iVoxelSuperSampling;

	void initializeFromProjector();
};

// inline functions
inline std::string CCudaCglsAlgorithm3D::description() const { return CCudaCglsAlgorithm3D::type; };

} // end namespace

#endif

#endif
