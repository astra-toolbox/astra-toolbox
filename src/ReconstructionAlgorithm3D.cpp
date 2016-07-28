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

#include "astra/ReconstructionAlgorithm3D.h"

#include "astra/AstraObjectManager.h"

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
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("ReconstructionAlgorithm3D", this, _cfg);

	XMLNode node;
	int id;

	// projector
	node = _cfg.self.getSingleNode("ProjectorId");
	m_pProjector = 0;
	if (node) {
		id = node.getContentInt();
		m_pProjector = CProjector3DManager::getSingleton().get(id);
		if (!m_pProjector) {
			// TODO: Report
		}
	}
	CC.markNodeParsed("ProjectorId");

	// sinogram data
	node = _cfg.self.getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction3D", "No ProjectionDataId tag specified.");
	id = node.getContentInt();
	m_pSinogram = dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("ProjectionDataId");

	// reconstruction data
	node = _cfg.self.getSingleNode("ReconstructionDataId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction3D", "No ReconstructionDataId tag specified.");
	id = node.getContentInt();
	m_pReconstruction = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("ReconstructionDataId");

	// fixed mask
	if (_cfg.self.hasOption("ReconstructionMaskId")) {
		m_bUseReconstructionMask = true;
		id = _cfg.self.getOptionInt("ReconstructionMaskId");
		m_pReconstructionMask = dynamic_cast<CFloat32VolumeData3D*>(CData3DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("ReconstructionMaskId");

	// fixed mask
	if (_cfg.self.hasOption("SinogramMaskId")) {
		m_bUseSinogramMask = true;
		id = _cfg.self.getOptionInt("SinogramMaskId");
		m_pSinogramMask = dynamic_cast<CFloat32ProjectionData3D*>(CData3DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("SinogramMaskId");

	// Constraints - NEW
	if (_cfg.self.hasOption("MinConstraint")) {
		m_bUseMinConstraint = true;
		m_fMinValue = _cfg.self.getOptionNumerical("MinConstraint", 0.0f);
		CC.markOptionParsed("MinConstraint");
	} else {
		// Constraint - OLD
		m_bUseMinConstraint = _cfg.self.getOptionBool("UseMinConstraint", false);
		CC.markOptionParsed("UseMinConstraint");
		if (m_bUseMinConstraint) {
			m_fMinValue = _cfg.self.getOptionNumerical("MinConstraintValue", 0.0f);
			CC.markOptionParsed("MinConstraintValue");
		}
	}
	if (_cfg.self.hasOption("MaxConstraint")) {
		m_bUseMaxConstraint = true;
		m_fMaxValue = _cfg.self.getOptionNumerical("MaxConstraint", 255.0f);
		CC.markOptionParsed("MaxConstraint");
	} else {
		// Constraint - OLD
		m_bUseMaxConstraint = _cfg.self.getOptionBool("UseMaxConstraint", false);
		CC.markOptionParsed("UseMaxConstraint");
		if (m_bUseMaxConstraint) {
			m_fMaxValue = _cfg.self.getOptionNumerical("MaxConstraintValue", 0.0f);
			CC.markOptionParsed("MaxConstraintValue");
		}
	}

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

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CReconstructionAlgorithm3D::getInformation() 
{
	map<string, boost::any> res;
	res["ProjectorId"] = getInformation("ProjectorId");
	res["ProjectionDataId"] = getInformation("ProjectionDataId");
	res["ReconstructionDataId"] = getInformation("ReconstructionDataId");
	res["UseMinConstraint"] = getInformation("UseMinConstraint");
	res["MinConstraintValue"] = getInformation("MinConstraintValue");
	res["UseMaxConstraint"] = getInformation("UseMaxConstraint");
	res["MaxConstraintValue"] = getInformation("MaxConstraintValue");
	res["ReconstructionMaskId"] = getInformation("ReconstructionMaskId");
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
};

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CReconstructionAlgorithm3D::getInformation(std::string _sIdentifier) 
{
	if (_sIdentifier == "UseMinConstraint")		{ return m_bUseMinConstraint ? string("yes") : string("no"); }
	if (_sIdentifier == "MinConstraintValue")	{ return m_fMinValue; }
	if (_sIdentifier == "UseMaxConstraint")		{ return m_bUseMaxConstraint ? string("yes") : string("no"); }
	if (_sIdentifier == "MaxConstraintValue")	{ return m_fMaxValue; }
#if 0
	if (_sIdentifier == "ProjectorId")	{ 
		int iIndex = CProjector3DManager::getSingleton().getIndex(m_pProjector);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
#endif
	if (_sIdentifier == "ProjectionDataId") {
		int iIndex = CData3DManager::getSingleton().getIndex(m_pSinogram);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	} 
	if (_sIdentifier == "ReconstructionDataId") {
		int iIndex = CData3DManager::getSingleton().getIndex(m_pReconstruction);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	if (_sIdentifier == "ReconstructionMaskId") {
		if (!m_bUseReconstructionMask) return string("not used");
		int iIndex = CData3DManager::getSingleton().getIndex(m_pReconstructionMask);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	return CAlgorithm::getInformation(_sIdentifier);
};
//----------------------------------------------------------------------------------------

} // namespace astra
