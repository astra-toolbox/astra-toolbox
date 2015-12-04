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

#ifndef _INC_ASTRA_CUDASIRTALGORITHM3D
#define _INC_ASTRA_CUDASIRTALGORITHM3D

#include "Globals.h"
#include "Config.h"

#include "Algorithm.h"

#include "Float32ProjectionData3DMemory.h"
#include "Float32VolumeData3DMemory.h"
#include "ReconstructionAlgorithm3D.h"

#ifdef ASTRA_CUDA

namespace astra {

class AstraSIRT3d;

/**
 * \brief
 * This class contains the 3D implementation of the SIRT (Simultaneous Iterative Reconstruction Technique) algorithm.
 *
 * The update step of pixel \f$v_j\f$ for iteration \f$k\f$ is given by:
 * \f[
 *	v_j^{(k+1)} = v_j^{(k)} + \alpha \sum_{i=1}^{M} \left( \frac{w_{ij}\left( p_i - \sum_{r=1}^{N} w_{ir}v_r^{(k)}\right)}{\sum_{k=1}^{N} w_{ik}} \right) \frac{1}{\sum_{l=1}^{M}w_{lj}}
 * \f]
 *
 * \par XML Configuration
 *
 * \par MATLAB example
 * \astra_code{
 *
 * }
 *
 * \par References
 * [1] "Computational Analysis and Improvement of SIRT", J. Gregor, T. Benson, IEEE Transactions on Medical Imaging, Vol. 22, No. 7, July 2008.
 */
class _AstraExport CCudaSirtAlgorithm3D : public CReconstructionAlgorithm3D {

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
	CCudaSirtAlgorithm3D();

	/** Constructor with initialization.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pProjectionData	ProjectionData3D object containing the projection data.
	 * @param _pReconstruction	VolumeData3D object for storing the reconstructed volume.
	 */
	CCudaSirtAlgorithm3D(CProjector3D* _pProjector, 
	                     CFloat32ProjectionData3DMemory* _pProjectionData, 
	                     CFloat32VolumeData3DMemory* _pReconstruction);
	
	/** Copy constructor.
	 */
	CCudaSirtAlgorithm3D(const CCudaSirtAlgorithm3D&);

	/** Destructor.
	 */
	virtual ~CCudaSirtAlgorithm3D();

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
					CFloat32ProjectionData3DMemory* _pSinogram, 
					CFloat32VolumeData3DMemory* _pReconstruction);

	/** Get all information parameters
	 *
	 * @return map with all boost::any object
	 */
	virtual map<string,boost::any> getInformation();

	/** Get a single piece of information represented as a boost::any
	 *
	 * @param _sIdentifier identifier string to specify which piece of information you want
	 * @return boost::any object
	 */
	virtual boost::any getInformation(std::string _sIdentifier);

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


	virtual void signalAbort();

	/** Get the norm of the residual image.
	 *  Only a few algorithms support this method.
	 *
	 * @param _fNorm if supported, the norm is returned here
	 * @return true if this operation is supported
	 */
	virtual bool getResidualNorm(float32& _fNorm);
	
	/*
	 * The required MPI code is implemented for this function / algorithm
	 * 
	*/	
	bool isMPICapable() {return true;}

protected:

	AstraSIRT3d* m_pSirt;

	int m_iGPUIndex;
	bool m_bAstraSIRTInit;
	int m_iDetectorSuperSampling;
	int m_iVoxelSuperSampling;

	void initializeFromProjector();
};

// inline functions
inline std::string CCudaSirtAlgorithm3D::description() const { return CCudaSirtAlgorithm3D::type; };

} // end namespace

#endif

#endif
