/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

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
*/

#ifndef _INC_ASTRA_RECONSTRUCTIONALGORITHM2D
#define _INC_ASTRA_RECONSTRUCTIONALGORITHM2D

#include "Globals.h"
#include "Config.h"

#include "Algorithm.h"

#include "Projector2D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"


namespace astra {

/**
 * This is a base class for the different implementations of 2D reconstruction algorithms.
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
 */
class _AstraExport CReconstructionAlgorithm2D : public CAlgorithm {

public:

	/** Default constructor, containing no code.
	 */
	CReconstructionAlgorithm2D();
	
	/** Destructor.
	 */
	virtual ~CReconstructionAlgorithm2D();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
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

	/** Add a min/max constraint to the reconstruction process
	 *
	 * @param _bUseMin		True if the algorithm should use a min constraint.
	 * @param _fMinValue	Lower value to clip pixel values to.
	 * @param _bUseMax		True if the algorithm should use a max constraint.
	 * @param _fMaxValue	Upper value to clip pixel values to.
	 */
	void setConstraints(bool _bUseMin, float32 _fMinValue, bool _bUseMax, float32 _fMaxValue);

	/** Set a fixed reconstruction mask. A pixel will only be used in the reconstruction if the 
	 * corresponding value in the mask is 1.
	 *
	 * @param _pMask Volume Data object containing fixed reconstruction mask
	 * @param _bEnable enable the use of this mask
	 */
	void setReconstructionMask(CFloat32VolumeData2D* _pMask, bool _bEnable = true);

	/** Set a fixed sinogram mask. A detector value will only be used in the reconstruction if the 
	 * corresponding value in the mask is 1.
	 *
	 * @param _pMask Projection Data object containing fixed sinogram mask
	 * @param _bEnable enable the use of this mask
	 */
	void setSinogramMask(CFloat32ProjectionData2D* _pMask, bool _bEnable = true);

	/** Get projector object
	 *
	 * @return projector
	 */
	CProjector2D* getProjector() const;

	/** Get sinogram data object
	 *
	 * @return sinogram data object
	 */
	CFloat32ProjectionData2D* getSinogram() const;

	/** Get Reconstructed Data
	 *
	 * @return reconstruction
	 */
	CFloat32VolumeData2D* getReconstruction() const;

	/** Get Fixed Reconstruction Mask
	 *
	 * @return fixed reconstruction mask
	 */
	CFloat32VolumeData2D* getReconstructionMask() const;

	/** Perform a number of iterations.
	 *
	 * @param _iNrIterations amount of iterations to perform.
	 */
	virtual void run(int _iNrIterations = 0) = 0;

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
	virtual bool getResidualNorm(float32& _fNorm) { return false; }

protected:
	
	/** Check this object.
	 *
	 * @return object initialized
	 */
	bool _check();

	/** Initial clearing. Only to be used by constructors.
	 */
	void _clear();

	//< Projector object.
	CProjector2D* m_pProjector;
	//< ProjectionData2D object containing the sinogram.
	CFloat32ProjectionData2D* m_pSinogram;
	//< VolumeData2D object for storing the reconstruction volume.
	CFloat32VolumeData2D* m_pReconstruction;
	
	//< Use minimum value constraint?
	bool m_bUseMinConstraint;
	//< Minimum value constraint.
	float32 m_fMinValue;
	//< Use maximum value constraint?
	bool m_bUseMaxConstraint;
	//< Maximum value constraint.
	float32 m_fMaxValue;

	//< Dataobject containing fixed reconstruction mask (0 = don't reconstruct)
	CFloat32VolumeData2D* m_pReconstructionMask;
	//< Use the fixed reconstruction mask?
	bool m_bUseReconstructionMask;

	//< Dataobject containing fixed reconstruction mask (0 = don't reconstruct)
	CFloat32ProjectionData2D* m_pSinogramMask;
	//< Use the fixed reconstruction mask?
	bool m_bUseSinogramMask;


	//< Specify if initialize/check should check for a valid Projector
	virtual bool requiresProjector() const { return true; }
};

// inline functions
inline std::string CReconstructionAlgorithm2D::description() const { return "3D Reconstruction Algorithm"; };
inline CProjector2D* CReconstructionAlgorithm2D::getProjector() const { return m_pProjector; }
inline CFloat32ProjectionData2D* CReconstructionAlgorithm2D::getSinogram() const { return m_pSinogram; }
inline CFloat32VolumeData2D* CReconstructionAlgorithm2D::getReconstruction() const { return m_pReconstruction; }
inline CFloat32VolumeData2D* CReconstructionAlgorithm2D::getReconstructionMask() const { return m_pReconstructionMask; }

} // end namespace

#endif
