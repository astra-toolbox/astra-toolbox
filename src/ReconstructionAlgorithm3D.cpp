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

#include "astra/ReconstructionAlgorithm3D.h"

#include "astra/AstraObjectManager.h"
#include "astra/Logging.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CReconstructionAlgorithm3D::CReconstructionAlgorithm3D() 
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bUseMinConstraint = false;
	m_fMinValue = 0.0f;
	m_bUseMaxConstraint = false;
	m_fMaxValue = 0.0f;
	m_bUseReconstructionMask = false;
	m_pReconstructionMask = NULL;
	m_bUseSinogramMask = false;
	m_pSinogramMask = NULL;
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CReconstructionAlgorithm3D::~CReconstructionAlgorithm3D() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CReconstructionAlgorithm3D::_clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bUseMinConstraint = false;
	m_fMinValue = 0.0f;
	m_bUseMaxConstraint = false;
	m_fMaxValue = 0.0f;
	m_bUseReconstructionMask = false;
	m_pReconstructionMask = NULL;
	m_bUseSinogramMask = false;
	m_pSinogramMask = NULL;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CReconstructionAlgorithm3D::clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bUseMinConstraint = false;
	m_fMinValue = 0.0f;
	m_bUseMaxConstraint = false;
	m_fMaxValue = 0.0f;
	m_bUseReconstructionMask = false;
	m_pReconstructionMask = NULL;
	m_bUseSinogramMask = false;
	m_pSinogramMask = NULL;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CReconstructionAlgorithm3D::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("ReconstructionAlgorithm3D", this, _cfg);

	// projector
	m_pProjector = 0;
	int id = -1;
	if (CR.has("ProjectorId")) {
		CR.getID("ProjectorId", id);
		m_pProjector = CProjector3DManager::getSingleton().get(id);
		if (!m_pProjector) {
			ASTRA_WARN("Optional parameter ProjectorId is not a valid id");
		}
	}

	bool ok = true;

	// sinogram data
	ok &= CR.getRequiredID("ProjectionDataId", id);
	m_pSinogram = dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(id));

	// reconstruction data
	ok &= CR.getRequiredID("ReconstructionDataId", id);
	m_pReconstruction = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));

	if (!ok)
		return false;

	// fixed mask
	if (CR.getOptionID("ReconstructionMaskId", id)) {
		m_bUseReconstructionMask = true;
		m_pReconstructionMask = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	}

	// fixed mask
	if (CR.getOptionID("SinogramMaskId", id)) {
		m_bUseSinogramMask = true;
		m_pSinogramMask = dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(id));
	}

	// Constraints - NEW
	if (CR.hasOption("MinConstraint")) {
		m_bUseMinConstraint = true;
		ok &= CR.getOptionNumerical("MinConstraint", m_fMinValue, 0.0f);
	} else {
		// Constraint - OLD
		ok &= CR.getOptionBool("UseMinConstraint", m_bUseMinConstraint, false);
		if (m_bUseMinConstraint) {
			ok &= CR.getOptionNumerical("MinConstraintValue", m_fMinValue, 0.0f);
			ASTRA_WARN("UseMinConstraint/MinConstraintValue are deprecated. Use \"MinConstraint\" instead.");
		}
	}
	if (CR.hasOption("MaxConstraint")) {
		m_bUseMaxConstraint = true;
		ok &= CR.getOptionNumerical("MaxConstraint", m_fMaxValue, 255.0f);
	} else {
		// Constraint - OLD
		ok &= CR.getOptionBool("UseMaxConstraint", m_bUseMaxConstraint, false);
		if (m_bUseMaxConstraint) {
			ok &= CR.getOptionNumerical("MaxConstraintValue", m_fMaxValue, 255.0f);
			ASTRA_WARN("UseMaxConstraint/MaxConstraintValue are deprecated. Use \"MaxConstraint\" instead.");
		}
	}
	if (!ok)
		return false;

	// return success
	return _check();
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CReconstructionAlgorithm3D::initialize(CProjector3D* _pProjector, 
							   CFloat32ProjectionData3D* _pSinogram, 
							   CFloat32VolumeData3D* _pReconstruction)
{
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	// return success
	return _check();
}

//---------------------------------------------------------------------------------------
// Set Constraints
void CReconstructionAlgorithm3D::setConstraints(bool _bUseMin, float32 _fMinValue, bool _bUseMax, float32 _fMaxValue)
{
	m_bUseMinConstraint = _bUseMin;
	m_fMinValue = _fMinValue;
	m_bUseMaxConstraint = _bUseMax;
	m_fMaxValue = _fMaxValue;
}

//----------------------------------------------------------------------------------------
// Set Fixed Reconstruction Mask
void CReconstructionAlgorithm3D::setReconstructionMask(CFloat32VolumeData3D* _pMask, bool _bEnable)
{
	// TODO: check geometry matches volume
	m_bUseReconstructionMask = _bEnable;
	m_pReconstructionMask = _pMask;
	if (m_pReconstructionMask == NULL) {
		m_bUseReconstructionMask = false;
	}
}

//----------------------------------------------------------------------------------------
// Set Fixed Sinogram Mask
void CReconstructionAlgorithm3D::setSinogramMask(CFloat32ProjectionData3D* _pMask, bool _bEnable)
{
	// TODO: check geometry matches sinogram
	m_bUseSinogramMask = _bEnable;
	m_pSinogramMask = _pMask;
	if (m_pSinogramMask == NULL) {
		m_bUseSinogramMask = false;
	}
}//----------------------------------------------------------------------------------------
// Check
bool CReconstructionAlgorithm3D::_check() 
{
	// check pointers
#if 0
	ASTRA_CONFIG_CHECK(m_pProjector, "Reconstruction3D", "Invalid Projector Object.");
#endif
	ASTRA_CONFIG_CHECK(m_pSinogram, "Reconstruction3D", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pReconstruction, "Reconstruction3D", "Invalid Reconstruction Data Object.");

	// check initializations
#if 0
	ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "Reconstruction3D", "Projector Object Not Initialized.");
#endif
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "Reconstruction3D", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isInitialized(), "Reconstruction3D", "Reconstruction Data Object Not Initialized.");

#if 0
	// check compatibility between projector and data classes
	ASTRA_CONFIG_CHECK(m_pSinogram->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "Reconstruction3D", "Projection Data not compatible with the specified Projector.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "Reconstruction3D", "Reconstruction Data not compatible with the specified Projector.");
#endif

	// success
	return true;
}

//----------------------------------------------------------------------------------------

} // namespace astra
