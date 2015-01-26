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

#ifndef _INC_ASTRA_FILTEREDBACKPROJECTION
#define _INC_ASTRA_FILTEREDBACKPROJECTION

#include "ReconstructionAlgorithm2D.h"

#include "Globals.h"

#include "Projector2D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"


namespace astra {

/**
 * \brief
 * This class contains the implementation of the filtered back projection (FBP)
 * reconstruction algorithm.
 *
 * \par XML Configuration
 * \astra_xml_item{ProjectorId, integer, Identifier of a projector as it is stored in the ProjectorManager.}
 * \astra_xml_item{VolumeDataId, integer, Identifier of the volume data object as it is stored in the DataManager.}
 * \astra_xml_item{ReconstructionDataId, integer, Identifier of the resulting projection data object as it is stored in the DataManager.}
 * \astra_xml_item_option{ProjectionIndex, integer, 0, Only reconstruct this specific projection angle. }

 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('FP');\n
 *		cfg.ProjectorId = proj_id;\n
 *		cfg.ReconstructionDataId = vol_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('run'\, alg_id);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 *
 */
class _AstraExport CFilteredBackProjectionAlgorithm : public CReconstructionAlgorithm2D {

protected:

	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - valid projector
	 * - valid data objects
	 * - projection order all within range
	 */
	virtual bool _check();

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;	

	/** Default constructor, containing no code.
	 */
	CFilteredBackProjectionAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CFilteredBackProjectionAlgorithm();

	/** Initialize class.
	 *
	 * @param _pProjector		Projector to use.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 * @return success
	 */
	bool initialize(CProjector2D* _pProjector, 
					CFloat32VolumeData2D* _pReconstruction, 
					CFloat32ProjectionData2D* _pSinogram);

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Clear this class.
	 */
	virtual void clear();

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

	/** Performs the filtering of the projection data.
	 *
	 * @param _pFilteredSinogram will contain filtered sinogram afterwards
	 */
	void performFiltering(CFloat32ProjectionData2D * _pFilteredSinogram);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

};

// inline functions
inline std::string CFilteredBackProjectionAlgorithm::description() const { return CFilteredBackProjectionAlgorithm::type; };

} // end namespace

#endif
