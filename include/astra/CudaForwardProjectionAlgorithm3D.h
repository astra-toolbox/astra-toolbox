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

#ifndef _INC_ASTRA_CUDAFORWARDPROJECTIONALGORITHM3D
#define _INC_ASTRA_CUDAFORWARDPROJECTIONALGORITHM3D

#include "Globals.h"

#include "Algorithm.h"

#include "Float32ProjectionData3DMemory.h"
#include "Float32VolumeData3DMemory.h"

#ifdef ASTRA_CUDA

namespace astra {

class CProjector3D;

class _AstraExport CCudaForwardProjectionAlgorithm3D : public CAlgorithm
{
public:

	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code.
	 */
	CCudaForwardProjectionAlgorithm3D();
	
	/** Destructor.
	 */
	virtual ~CCudaForwardProjectionAlgorithm3D();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pProjectionData	ProjectionData3D object for storing the projection data.
	 * @param _pReconstruction	VolumeData3D object containing the volume.
	 * @return initialization successful?
	 */
	bool initialize(CProjector3D* _pProjector, 
					CFloat32ProjectionData3DMemory* _pSinogram, 
					CFloat32VolumeData3DMemory* _pReconstruction,
					int _iGPUindex = -1, int _iDetectorSuperSampling = 1);


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

	/** Check this object.
	 *
	 * @return object initialized
	 */
	bool check();

	/**  
	 * Sets the index of the used GPU index: first GPU has index 0
	 *
	 * @param _iGPUIndex New GPU index.
	 */
	void setGPUIndex(int _iGPUIndex);

protected:
	CProjector3D* m_pProjector;
	CFloat32ProjectionData3DMemory* m_pProjections;
	CFloat32VolumeData3DMemory* m_pVolume;
	int m_iGPUIndex;
	int m_iDetectorSuperSampling;

	void initializeFromProjector();
};

// inline functions
inline std::string CCudaForwardProjectionAlgorithm3D::description() const { return CCudaForwardProjectionAlgorithm3D::type; };


}

#endif

#endif
