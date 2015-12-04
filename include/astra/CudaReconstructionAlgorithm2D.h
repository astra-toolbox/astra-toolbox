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

#ifndef _INC_ASTRA_CUDARECONSTRUCTIONALGORITHM2D
#define _INC_ASTRA_CUDARECONSTRUCTIONALGORITHM2D

#include "Globals.h"
#include "Config.h"

#include "ReconstructionAlgorithm2D.h"

#include "Projector2D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"

namespace astraCUDA {
class ReconAlgo;
}

namespace astra {

/**
 * This is a base class for the different CUDA implementations of 2D reconstruction algorithms.
 * They don't use a Projector, and share GPUIndex and DetectorSuperSampling options.
 *
 */
class _AstraExport CCudaReconstructionAlgorithm2D : public CReconstructionAlgorithm2D {

public:

	/** Default constructor, containing no code.
	 */
	CCudaReconstructionAlgorithm2D();
	
	/** Destructor.
	 */
	virtual ~CCudaReconstructionAlgorithm2D();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
	 *
	 * @param _pProjector		Projector Object. (Optional)
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 */
	virtual bool initialize(CProjector2D* _pProjector, 
	                        CFloat32ProjectionData2D* _pSinogram, 
	                        CFloat32VolumeData2D* _pReconstruction);


	/** Clear this class.
	 */
	virtual void clear();

	/** Get all information parameters.
	 *
	 * @return map with all boost::any object
	 */
	virtual map<string,boost::any> getInformation();

	/** Get a single piece of information.
	 *
	 * @param _sIdentifier identifier string to specify which piece of information you want
	 * @return boost::any object
	 */
	virtual boost::any getInformation(std::string _sIdentifier);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

	/** Get the norm of the residual image.
	 *  Only a few algorithms support this method.
	 *
	 * @param _fNorm if supported, the norm is returned here
	 * @return true if this operation is supported
	 */
	virtual bool getResidualNorm(float32& _fNorm);

	/**  
	 * Sets the index of the used GPU index: first GPU has index 0
	 *
	 * @param _iGPUIndex New GPU index.
	 */
	void setGPUIndex(int _iGPUIndex);

	/** Perform a number of iterations.
	 *
	 * @param _iNrIterations amount of iterations to perform.
	 */
	virtual void run(int _iNrIterations = 0);

	virtual void signalAbort();

protected:
	
	/** Check this object.
	 *
	 * @return object initialized
	 */
	bool _check();

	/** Initial clearing. Only to be used by constructors.
	 */
	void _clear();

	/** Set up geometry. For internal use only.
	 */
	bool setupGeometry();


	/** The internally used CUDA algorithm object
	 */ 
	astraCUDA::ReconAlgo *m_pAlgo;

	int m_iDetectorSuperSampling;
	int m_iPixelSuperSampling;
	int m_iGPUIndex;

	bool m_bAlgoInit;

	void initializeFromProjector();
	virtual bool requiresProjector() const { return false; }
};

// inline functions
inline std::string CCudaReconstructionAlgorithm2D::description() const { return "2D CUDA Reconstruction Algorithm"; };

} // end namespace

#endif
