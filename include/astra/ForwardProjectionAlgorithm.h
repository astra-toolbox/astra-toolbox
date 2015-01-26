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

#ifndef _INC_ASTRA_FORWARDPROJECTIONALGORITHM
#define _INC_ASTRA_FORWARDPROJECTIONALGORITHM

#include "Algorithm.h"

#include "Globals.h"

#include "Projector2D.h"
#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"

#include "DataProjector.h"

namespace astra {

/**
 * \brief
 * This class contains the implementation of an algorithm that creates a forward projection 
 * of a volume object and stores it into a sinogram.
 *
 * \par XML Configuration
 * \astra_xml_item{ProjectorId, integer, Identifier of a projector as it is stored in the ProjectorManager.}
 * \astra_xml_item{VolumeDataId, integer, Identifier of the volume data object as it is stored in the DataManager.}
 * \astra_xml_item{ProjectionDataId, integer, Identifier of the resulting projection data object as it is stored in the DataManager.}
 * \astra_xml_item_option{VolumeMaskId, integer, not used, Identifier of a volume data object that acts as a volume mask. 0 = don't use this pixel. 1 = use this pixel. }
 * \astra_xml_item_option{SinogramMaskId, integer, not used, Identifier of a projection data object that acts as a projection mask. 0 = don't use this ray. 1 = use this ray.}
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('FP');\n
 *		cfg.ProjectorId = proj_id;\n
 *		cfg.VolumeDataId = vol_id;\n
 *		cfg.ProjectionDataId = sino_id;\n
 *		alg_id = astra_mex_algorithm('create'\, cfg);\n
 *		astra_mex_algorithm('run'\, alg_id);\n
 *		astra_mex_algorithm('delete'\, alg_id);\n
 * }
 *
 */
class _AstraExport CForwardProjectionAlgorithm : public CAlgorithm {

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

	//< Projector object.
	CProjector2D* m_pProjector;
	//< ProjectionData2D object containing the sinogram.
	CFloat32ProjectionData2D* m_pSinogram;
	//< VolumeData2D object containing the phantom.
	CFloat32VolumeData2D* m_pVolume;

	// data projector
	astra::CDataProjectorInterface* m_pForwardProjector;

	// ray or voxel-driven projector code?
	bool m_bUseVoxelProjector;

	//< Dataobject containing fixed volume mask (0 = don't project)
	CFloat32VolumeData2D* m_pVolumeMask;
	//< Use the fixed reconstruction mask?
	bool m_bUseVolumeMask;

	//< Dataobject containing fixed reconstruction mask (0 = don't project)
	CFloat32ProjectionData2D* m_pSinogramMask;
	//< Use the fixed reconstruction mask?
	bool m_bUseSinogramMask;

public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;	

	/** Default constructor, containing no code.
	 */
	CForwardProjectionAlgorithm();

	/** Initializing constructor.
	 *
	 * @param _pProjector		Projector to use.
	 * @param _pVolume			VolumeData2D object containing the phantom to compute sinogram from		
	 * @param _pSinogram		ProjectionData2D object to store sinogram data in.
	 */
	CForwardProjectionAlgorithm(CProjector2D* _pProjector, 
								CFloat32VolumeData2D* _pVolume, 
								CFloat32ProjectionData2D* _pSinogram);

	/** Destructor.
	 */
	virtual ~CForwardProjectionAlgorithm();

	/** Clear this class.
	 */
	virtual void clear();

	/** Initialize class.
	 *
	 * @param _pProjector		Projector to use.
	 * @param _pVolume			VolumeData2D object containing the phantom to compute sinogram from		
	 * @param _pSinogram		ProjectionData2D object to store sinogram data in.
	 * @return success
	 */
	bool initialize(CProjector2D* _pProjector, 
					CFloat32VolumeData2D* _pVolume, 
					CFloat32ProjectionData2D* _pSinogram);

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

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

	/** Set a fixed reconstruction mask. A pixel will only be used in the reconstruction if the 
	 * corresponding value in the mask is 1.
	 *
	 * @param _pMask Volume Data object containing fixed reconstruction mask
	 * @param _bEnable enable the use of this mask
	 */
	void setVolumeMask(CFloat32VolumeData2D* _pMask, bool _bEnable = true);

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

	/** Get volume data object
	 *
	 * @return volume data object
	 */
	CFloat32VolumeData2D* getVolume() const;

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
inline std::string CForwardProjectionAlgorithm::description() const { return CForwardProjectionAlgorithm::type; };
inline CProjector2D* CForwardProjectionAlgorithm::getProjector() const { return m_pProjector; }
inline CFloat32ProjectionData2D* CForwardProjectionAlgorithm::getSinogram() const { return m_pSinogram; }
inline CFloat32VolumeData2D* CForwardProjectionAlgorithm::getVolume() const { return m_pVolume; }

} // end namespace

#endif
