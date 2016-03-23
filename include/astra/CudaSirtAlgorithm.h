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

#ifndef _INC_ASTRA_CUDASIRTALGORITHM2
#define _INC_ASTRA_CUDASIRTALGORITHM2

#include "Globals.h"
#include "Config.h"

#include "CudaReconstructionAlgorithm2D.h"

#ifdef ASTRA_CUDA

namespace astra {

/**
 * \brief
 * This class contains a GPU implementation of the SIRT (Simultaneous Iterative Reconstruction Technique) algorithm.
 *
 * The update step of pixel \f$v_j\f$ for iteration \f$k\f$ is given by:
 * \f[
 *	v_j^{(k+1)} = v_j^{(k)} + \alpha \sum_{i=1}^{M} \left( \frac{w_{ij}\left( p_i - \sum_{r=1}^{N} w_{ir}v_r^{(k)}\right)}{\sum_{k=1}^{N} w_{ik}} \right) \frac{1}{\sum_{l=1}^{M}w_{lj}}
 * \f]
 *
 * \par XML Configuration
 * \astra_xml_item{ProjectionGeometry, integer, Geometry of the projection data.}
 * \astra_xml_item{VolumeGeometry, integer, Geometry of the volume data.}
 * \astra_xml_item{ProjectionDataId, integer, Identifier of a projection data object as it is stored in the DataManager.}
 * \astra_xml_item{ReconstructionDataId, integer, Identifier of a volume data object as it is stored in the DataManager.}
 * \astra_xml_item_option{ReconstructionMaskId, integer, not used, Identifier of a volume data object that acts as a reconstruction mask. 0 = reconstruct on this pixel. 1 = don't reconstruct on this pixel.}
 * \astra_xml_item_option{Relaxation, float, 1, Relaxation parameter.}
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('SIRT_CUDA2');\n
 *		cfg.ProjectionGeometry = proj_geom;\n
 *		cfg.VolumeGeometry = vol_geom;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		cfg.ReconstructionDataId = recon_id;\n
 *		cfg.option.ReconstructionMaskId = mask_id;\n
 *		cfg.option.Relaxation = 1.0;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('iterate'\, alg_id\, 10);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 *
 * \par References
 * [1] "Computational Analysis and Improvement of SIRT", J. Gregor, T. Benson, IEEE Transactions on Medical Imaging, Vol. 22, No. 7, July 2008.
 */
class _AstraExport CCudaSirtAlgorithm : public CCudaReconstructionAlgorithm2D
{
	
public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code.
	 */
	CCudaSirtAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CCudaSirtAlgorithm();

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
	bool initialize(CProjector2D* _pProjector,
	                CFloat32ProjectionData2D* _pSinogram,
	                CFloat32VolumeData2D* _pReconstruction);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

protected:
	CFloat32VolumeData2D* m_pMinMask;
	CFloat32VolumeData2D* m_pMaxMask;

	/** Relaxation factor
	 */
	float m_fLambda;

	virtual void initCUDAAlgorithm();
};

// inline functions
inline std::string CCudaSirtAlgorithm::description() const { return CCudaSirtAlgorithm::type; };

} // end namespace

#endif // ASTRA_CUDA

#endif 
