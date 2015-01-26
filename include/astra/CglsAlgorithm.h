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

#ifndef _INC_ASTRA_CGLSALGORITHM
#define _INC_ASTRA_CGLSALGORITHM

#include "Globals.h"
#include "Config.h"

#include "Algorithm.h"
#include "ReconstructionAlgorithm2D.h"

#include "Projector2D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"

#include "DataProjector.h"

namespace astra {

/**
 * \brief
 * This class contains the implementation of the CGLS (Conguent Gradient Algorithm) algorithm.
 *
 * \par XML Configuration
 * \astra_xml_item{ProjectorId, integer, Identifier of a projector as it is stored in the ProjectorManager.}
 * \astra_xml_item{ProjectionDataId, integer, Identifier of a projection data object as it is stored in the DataManager.}
 * \astra_xml_item{ReconstructionDataId, integer, Identifier of a volume data object as it is stored in the DataManager.}
 * \astra_xml_item_option{ReconstructionMaskId, integer, not used, Identifier of a volume data object that acts as a reconstruction mask. 0 = reconstruct on this pixel. 1 = don't reconstruct on this pixel.}
 * \astra_xml_item_option{SinogramMaskId, integer, not used, Identifier of a projection data object that acts as a projection mask. 0 = reconstruct using this ray. 1 = don't use this ray while reconstructing.}
 * \astra_xml_item_option{UseMinConstraint, bool, false, Use minimum value constraint.}
 * \astra_xml_item_option{MinConstraintValue, float, 0, Minimum constraint value.}
 * \astra_xml_item_option{UseMaxConstraint, bool, false, Use maximum value constraint.}
 * \astra_xml_item_option{MaxConstraintValue, float, 255, Maximum constraint value.}
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('CGLS');\n
 *		cfg.ProjectorId = proj_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		cfg.ReconstructionDataId = recon_id;\n
 *		cfg.option.MaskId = mask_id;\n
 *		cfg.option.UseMinConstraint = 'yes';\n 
 *		cfg.option.UseMaxConstraint = 'yes';\n
 *		cfg.option.MaxConstraintValue = 1024;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('iterate'\, alg_id\, 10);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 *
 */
class _AstraExport CCglsAlgorithm : public CReconstructionAlgorithm2D {

protected:

	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - valid projector
	 * - valid data objects
	 */
	virtual bool _check();

	CFloat32ProjectionData2D* r;
	CFloat32ProjectionData2D* w;
	CFloat32VolumeData2D* z;
	CFloat32VolumeData2D* p;

	float32 alpha;
	float32 beta;
	float32 gamma;


	int m_iIteration;

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code. 
	 */
	CCglsAlgorithm();

	/** Default constructor
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 */
	CCglsAlgorithm(CProjector2D* _pProjector, 
				   CFloat32ProjectionData2D* _pSinogram, 
				   CFloat32VolumeData2D* _pReconstruction);

	/** Destructor. 
	 */
	virtual ~CCglsAlgorithm();

	/** Clear this class.
	 */
	virtual void clear();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return Initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 * @return Initialization successful?
	 */
	bool initialize(CProjector2D* _pProjector, 
					CFloat32ProjectionData2D* _pSinogram, 
					CFloat32VolumeData2D* _pReconstruction);

	/** Get all information parameters.
	 *
	 * @return Map with all available identifier strings and their values.
	 */
	virtual map<string,boost::any> getInformation();

	/** Get a single piece of information represented as a boost::any
	 *
	 * @param _sIdentifier Identifier string to specify which piece of information you want.
	 * @return One piece of information.
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

};

// inline functions
inline std::string CCglsAlgorithm::description() const { return CCglsAlgorithm::type; };


} // end namespace

#endif
