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

#ifndef _INC_ASTRA_SARTALGORITHM
#define _INC_ASTRA_SARTALGORITHM

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
 * This class contains the implementation of the SART (Simultaneous Algebraic Reconstruction Technique) algorithm.
 *
 * The update step of pixel \f$v_j\f$ for projection \f$phi\f$ and iteration \f$k\f$ is given by:
 * \f[
 *	v_j^{(k+1)} = v_j^{(k)} + \lambda \frac{\sum_{p_i \in P_\phi} \left(  \frac{p_i - \sum_{r=1}^{N} w_{ir}v_r^{(k)}} {\sum_{r=1}^{N}w_{ir} }    \right)} {\sum_{p_i \in P_\phi}w_{ij}}
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
 * \astra_xml_item_option{ProjectionOrder, string, "sequential", the order in which the projections are updated. 'sequential', 'random' or 'custom'}
 * \astra_xml_item_option{ProjectionOrderList, vector of float, not used, if ProjectionOrder='custom': use this order.}
 * \astra_xml_item_option{Relaxation, float, 1, The relaxation parameter.}
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('SART');\n
 *		cfg.ProjectorId = proj_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		cfg.ReconstructionDataId = recon_id;\n
 *		cfg.option.MaskId = mask_id;\n
 *		cfg.option.UseMinConstraint = 'yes';\n 
 *		cfg.option.UseMaxConstraint = 'yes';\n
 *		cfg.option.MaxConstraintValue = 1024;\n
 *		cfg.option.ProjectionOrder = 'custom';\n
 *		cfg.option.ProjectionOrderList = randperm(100);\n
 *		cfg.option.Relaxation = 1.0;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('iterate'\, alg_id\, 10);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 */
class _AstraExport CSartAlgorithm : public CReconstructionAlgorithm2D {

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

	// temporary data objects
	CFloat32ProjectionData2D* m_pTotalRayLength;
	CFloat32VolumeData2D* m_pTotalPixelWeight;
	CFloat32ProjectionData2D* m_pDiffSinogram;

	int m_iIterationCount;

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;	
	
	/** Default constructor, containing no code.
	 */
	CSartAlgorithm();
	
	/** Constructor.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 */
	CSartAlgorithm(CProjector2D* _pProjector, 
				   CFloat32ProjectionData2D* _pSinogram, 
				   CFloat32VolumeData2D* _pReconstruction);

	/** Constructor.
	 *
	 * @param _pProjector			Projector Object.
	 * @param _pSinogram			ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction		VolumeData2D object for storing the reconstructed volume.
	 * @param _piProjectionOrder	array containing a projection order.
	 * @param _iProjectionCount		number of elements in _piProjectionOrder.
	 */
	CSartAlgorithm(CProjector2D* _pProjector, 
				   CFloat32ProjectionData2D* _pSinogram, 
				   CFloat32VolumeData2D* _pReconstruction,
				   int* _piProjectionOrder, 
				   int _iProjectionCount);

	/** Destructor.
	 */
	virtual ~CSartAlgorithm();
	
	/** Clear this class.
	 */
	virtual void clear();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class, no optionals, use sequential order.
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 * @return initialization successful?	 
	 */
	virtual bool initialize(CProjector2D* _pProjector, 
							CFloat32ProjectionData2D* _pSinogram, 
							CFloat32VolumeData2D* _pReconstruction);

	/** Initialize class, use custom order.
	 *
	 * @param _pProjector			Projector Object.
	 * @param _pSinogram			ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction		VolumeData2D object for storing the reconstructed volume.
	 * @param _piProjectionOrder	array containing a projection order.
	 * @param _iProjectionCount		number of elements in _piProjectionOrder.
	 * @return initialization successful?	 
	 */
	virtual bool initialize(CProjector2D* _pProjector, 
							CFloat32ProjectionData2D* _pSinogram, 
							CFloat32VolumeData2D* _pReconstruction,
							int* _piProjectionOrder, 
							int _iProjectionCount);

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

	/** Perform a number of iterations.  Each iteration is a forward and backprojection of 
	 * a single projection index.
	 *
	 * @param _iNrIterations amount of iterations to perform.
	 */
	virtual void run(int _iNrIterations = 1);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

protected:


	//< Order of the projections.
	int* m_piProjectionOrder;
	//< Number of projections specified in m_piProjectionOrder.
	int m_iProjectionCount;
	//< Current index in the projection order array.
	int m_iCurrentProjection;

	//< Relaxation parameter
	float m_fLambda;
};

// inline functions
inline std::string CSartAlgorithm::description() const { return CSartAlgorithm::type; };


} // end namespace

#endif
