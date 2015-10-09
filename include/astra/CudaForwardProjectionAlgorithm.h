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

#ifndef _INC_ASTRA_CUDAFORWARDPROJECTIONALGORITHM2
#define _INC_ASTRA_CUDAFORWARDPROJECTIONALGORITHM2

#include "Globals.h"

#include "Algorithm.h"

#ifdef ASTRA_CUDA

namespace astra {

class CProjector2D;
class CProjectionGeometry2D;
class CFloat32ProjectionData2D;
class CFloat32VolumeData2D;

/**
 * \brief
 * This class contains a GPU implementation of an algorithm that creates a forward projection 
 * of a volume object and stores it into a sinogram.
 *
 * \par XML Configuration
 * \astra_xml_item{VolumeGeometry, integer, Geometry of the volume data.}
 * \astra_xml_item{ProjectionGeometry, integer, Geometry of the projection data.}
 * \astra_xml_item{VolumeDataId, integer, Identifier of the volume data object as it is stored in the DataManager.}
 * \astra_xml_item{ProjectionDataId, integer, Identifier of the resulting projection data object as it is stored in the DataManager.}
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('FP_CUDA2');\n
 *		cfg.VolumeGeometry = vol_geom;\n
 *		cfg.ProjectionGeometry = proj_geom;\n
 *		cfg.VolumeDataId = vol_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('run'\, alg_id);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 *
 */
class _AstraExport CCudaForwardProjectionAlgorithm : public CAlgorithm
{	
public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code.
	 */
	CCudaForwardProjectionAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CCudaForwardProjectionAlgorithm();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
	 *
	 * @param _pProjector		Projector2D object. (Optional)
	 * @param _pVolume				VolumeData2D object containing the phantom to compute sinogram from		
	 * @param _pSinogram			ProjectionData2D object to store sinogram data in.
	 * @return success
	 */
	bool initialize(CProjector2D* _pProjector,
	                CFloat32VolumeData2D* _pVolume,
	                CFloat32ProjectionData2D* _pSinogram);


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

	CFloat32VolumeData2D* getVolume() { return m_pVolume; }
	CFloat32ProjectionData2D* getSinogram() { return m_pSinogram; }

	/**  
	 * Sets the index of the used GPU index: first GPU has index 0
	 *
	 * @param _iGPUIndex New GPU index.
	 */
	void setGPUIndex(int _iGPUIndex);

protected:
	//< Optional Projector2D object
	CProjector2D* m_pProjector;

	//< ProjectionData2D object containing the sinogram.
	CFloat32ProjectionData2D* m_pSinogram;
	//< VolumeData2D object containing the phantom.
	CFloat32VolumeData2D* m_pVolume;

	//< Index of GPU to use
	int m_iGPUIndex;
	//< Number of rays per detector element
	int m_iDetectorSuperSampling;

	void initializeFromProjector();
};

// inline functions
inline std::string CCudaForwardProjectionAlgorithm::description() const { return CCudaForwardProjectionAlgorithm::type; };

} // end namespace

#endif // ASTRA_CUDA

#endif
