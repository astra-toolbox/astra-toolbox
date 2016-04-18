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

#ifndef _INC_ASTRA_SIRTALGORITHM
#define _INC_ASTRA_SIRTALGORITHM

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
 * This class contains the implementation of the SIRT (Simultaneous Iterative Reconstruction Technique) algorithm.
 *
 * The update step of pixel \f$v_j\f$ for iteration \f$k\f$ is given by:
 * \f[
 *	v_j^{(k+1)} = v_j^{(k)} + \lambda \sum_{i=1}^{M} \left( \frac{w_{ij}\left( p_i - \sum_{r=1}^{N} w_{ir}v_r^{(k)}\right)}{\sum_{k=1}^{N} w_{ik}} \right) \frac{1}{\sum_{l=1}^{M}w_{lj}}
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
 *
 * \par XML Example
 * \astra_code{
 *		&lt;Algorithm type="SIRT"&gt;\n
 *		&lt;ProjectorID&gt;proj_id&lt;/ProjectorID&gt;\n
 *		&lt;ProjectionDataId&gt;sino_id&lt;/ProjectionDataId&gt;\n
 *		&lt;ReconstructionDataId&gt;recon_id&lt;/ReconstructionDataId&gt;\n
 *		&lt;Option key="ReconstructionMaskId" value="3"/&gt;\n
 *		&lt;Option key="SinogramMaskId" value="4"/&gt;\n
 *		&lt;Option key="UseMinConstraint" value="yes"/&gt;\n
 *		&lt;Option key="UseMaxConstraint" value="yes"/&gt;\n
 *		&lt;Option key="MaxConstraintValue" value="1024"/&gt;\n
 *		&lt;Option key="Relaxation" value="1"/&gt;\n
 *		&lt;/Algorithm&gt;
 * }
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('SIRT');\n
 *		cfg.ProjectorId = proj_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		cfg.ReconstructionDataId = recon_id;\n
 *		cfg.option.SinogramMaskId = smask_id;\n
 *		cfg.option.ReconstructionMaskId = mask_id;\n
 *		cfg.option.UseMinConstraint = 'yes';\n 
 *		cfg.option.UseMaxConstraint = 'yes';\n
 *		cfg.option.MaxConstraintValue = 1024;\n
 *		cfg.option.Relaxation = 1.0;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('iterate'\, alg_id\, 10);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 *
 * \par References
 * [1] "Computational Analysis and Improvement of SIRT", J. Gregor, T. Benson, IEEE Transactions on Medical Imaging, Vol. 22, No. 7, July 2008.
 */
class _AstraExport CSirtAlgorithm : public CReconstructionAlgorithm2D {

protected:

	/** Init stuff
	 */
	virtual void _init();

	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - valid projector
	 * - valid data objects
	 */
	virtual bool _check();

	/** Temporary data object for storing the total ray lengths
	 */
	CFloat32ProjectionData2D* m_pTotalRayLength;
	
	/** Temporary data object for storing the total pixel weigths
	 */
	CFloat32VolumeData2D* m_pTotalPixelWeight;
	
	/** Temporary data object for storing the difference between the forward projected
	 * reconstruction, and the measured projection data
	 */
	CFloat32ProjectionData2D* m_pDiffSinogram;

	/** Temporary data object for storing volume data
	*/
	CFloat32VolumeData2D* m_pTmpVolume;

	/** The number of performed iterations
	 */
	int m_iIterationCount;

	/** Relaxation parameter
	 */
	float m_fLambda;

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code. 
	 */
	CSirtAlgorithm();

	/** Default constructor
	 *
	 * @param _pProjector		Projector Object.
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 */
	CSirtAlgorithm(CProjector2D* _pProjector, 
				   CFloat32ProjectionData2D* _pSinogram, 
				   CFloat32VolumeData2D* _pReconstruction);

	/** Destructor. 
	 */
	virtual ~CSirtAlgorithm();

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
inline std::string CSirtAlgorithm::description() const { return CSirtAlgorithm::type; };


} // end namespace

#endif
