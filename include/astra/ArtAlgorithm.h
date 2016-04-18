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

#ifndef _INC_ASTRA_ARTALGORITHM
#define _INC_ASTRA_ARTALGORITHM

#include "Globals.h"
#include "Config.h"

#include "Algorithm.h"
#include "ReconstructionAlgorithm2D.h"

#include "Projector2D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"

namespace astra {

/**
 * This class contains the implementation of the ART (Algebraic Reconstruction Technique) algorithm.
 *
 * The update step of pixel \f$v_j\f$ for ray \f$i\f$ and iteration \f$k\f$ is given by:
 * \f[
 *	v_j^{(k+1)} = v_j^{(k)} + \lambda \frac{p_i - \sum_{r=1}^{N} w_{ir}v_r^{(k)}}{\sum_{k=1}^{N} w_{ik}^2} 
 * \f]
 * 
 * \par XML Configuration
 * \astra_xml_item{ProjectorId, integer, Identifier of a projector as it is stored in the ProjectorManager.}
 * \astra_xml_item{ProjectionDataId, integer, Identifier of a projection data object as it is stored in the DataManager.}
 * \astra_xml_item{ReconstructionDataId, integer, Identifier of a volume data object as it is stored in the DataManager.}
 * \astra_xml_item_option{ReconstructionMaskId, integer, not used, Identifier of a volume data object that acts as a reconstruction mask. 1 = reconstruct on this pixel. 0 = don't reconstruct on this pixel.}
 * \astra_xml_item_option{SinogramMaskId, integer, not used, Identifier of a projection data object that acts as a projection mask. 1 = reconstruct using this ray. 0 = don't use this ray while reconstructing.}
 * \astra_xml_item_option{UseMinConstraint, bool, false, Use minimum value constraint.}
 * \astra_xml_item_option{MinConstraintValue, float, 0, Minimum constraint value.}
 * \astra_xml_item_option{UseMaxConstraint, bool, false, Use maximum value constraint.}
 * \astra_xml_item_option{MaxConstraintValue, float, 255, Maximum constraint value.}
 * \astra_xml_item_option{Relaxation, float, 1, The relaxation factor.}
 * \astra_xml_item_option{RayOrder, string, "sequential", the order in which the rays are updated. 'sequential' or 'custom'}
 * \astra_xml_item_option{RayOrderList, n by 2 vector of float, not used, if RayOrder='custom': use this ray order.  Each row consist of a projection id and detector id.}
 * 
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('ART');\n
 *		cfg.ProjectorId = proj_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		cfg.ReconstructionDataId = recon_id;\n
 *		cfg.option.MaskId = mask_id;\n
 *		cfg.option.UseMinConstraint = 'yes';\n 
 *		cfg.option.UseMaxConstraint = 'yes';\n
 *		cfg.option.MaxConstraintValue = 1024;\n
 *		cfg.option.Relaxation = 0.7;\n
 *		cfg.option.RayOrder = 'custom';\n
 *		cfg.option.RayOrderList = [0\,0; 0\,2; 1\,0];\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('iterate'\, alg_id\, 1000);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 */
class _AstraExport CArtAlgorithm : public CReconstructionAlgorithm2D {

protected:

	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - no NULL pointers
	 * - all sub-objects are initialized properly
	 * - the projector is compatible with both data objects
	 * - the ray order list only contains valid values
	 */
	virtual bool _check();

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code.
	 */
	CArtAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CArtAlgorithm();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class, use sequential ray order.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 */
	bool initialize(CProjector2D* _pProjector, 
					CFloat32ProjectionData2D* _pSinogram, 
					CFloat32VolumeData2D* _pReconstruction);

	/** Clear this class.
	 */
	virtual void clear();

	/** Set the relaxation factor.
	 *
	 * @param _fLambda	Relaxation factor
	 */
	void setLambda(float32 _fLambda);

	/** Set the order in which the rays will be selected
	 *
	 * @param _piProjectionOrder	Order of the rays, the projections. (size should be _piRayCount)
	 * @param _piDetectorOrder		Order of the rays, the detectors. (size should be _piRayCount)
	 * @param _piRayCount			Number of rays in the two previous arrays.
	 */
	void setRayOrder(int* _piProjectionOrder, int* _piDetectorOrder, int _piRayCount);

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

protected:

	//< Relaxation Factor
	float32 m_fLambda;
	
	//< Order of the rays, the projections.
	int* m_piProjectionOrder;
	//< Order of the rays, the detectors.
	int* m_piDetectorOrder;
	//< Number of rays specified in the ray order arrays.
	int m_iRayCount;
	//< Current index in the ray order arrays.
	int m_iCurrentRay;
	
};

// inline functions
inline std::string CArtAlgorithm::description() const { return CArtAlgorithm::type; };


} // end namespace

#endif
