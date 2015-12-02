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

#include "astra/ReconstructionAlgorithm2D.h"

#include <boost/lexical_cast.hpp>

#include "astra/AstraObjectManager.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CReconstructionAlgorithm2D::CReconstructionAlgorithm2D() 
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CReconstructionAlgorithm2D::~CReconstructionAlgorithm2D() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CReconstructionAlgorithm2D::_clear()
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
void CReconstructionAlgorithm2D::clear()
{
	// Nothing to delete, so just _clear()
	_clear();
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CReconstructionAlgorithm2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("ReconstructionAlgorithm2D", this, _cfg);
	
	// projector
	XMLNode node = _cfg.self.getSingleNode("ProjectorId");
	if (requiresProjector()) {
		ASTRA_CONFIG_CHECK(node, "Reconstruction2D", "No ProjectorId tag specified.");
	}
	int id;
	if (node) {
		id = boost::lexical_cast<int>(node.getContent());
		m_pProjector = CProjector2DManager::getSingleton().get(id);
	} else {
		m_pProjector = 0;
	}
	CC.markNodeParsed("ProjectorId");

	// sinogram data
	node = _cfg.self.getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction2D", "No ProjectionDataId tag specified.");
	id = boost::lexical_cast<int>(node.getContent());
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("ProjectionDataId");

	// reconstruction data
	node = _cfg.self.getSingleNode("ReconstructionDataId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction2D", "No ReconstructionDataId tag specified.");
	id = boost::lexical_cast<int>(node.getContent());
	m_pReconstruction = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("ReconstructionDataId");

	// fixed mask
	if (_cfg.self.hasOption("ReconstructionMaskId")) {
		m_bUseReconstructionMask = true;
		id = boost::lexical_cast<int>(_cfg.self.getOption("ReconstructionMaskId"));
		m_pReconstructionMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
		ASTRA_CONFIG_CHECK(m_pReconstructionMask, "Reconstruction2D", "Invalid ReconstructionMaskId.");
	}
	CC.markOptionParsed("ReconstructionMaskId");

	// fixed mask
	if (_cfg.self.hasOption("SinogramMaskId")) {
		m_bUseSinogramMask = true;
		id = boost::lexical_cast<int>(_cfg.self.getOption("SinogramMaskId"));
		m_pSinogramMask = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
		ASTRA_CONFIG_CHECK(m_pSinogramMask, "Reconstruction2D", "Invalid SinogramMaskId.");
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
bool CReconstructionAlgorithm2D::initialize(CProjector2D* _pProjector, 
							   CFloat32ProjectionData2D* _pSinogram, 
							   CFloat32VolumeData2D* _pReconstruction)
{
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	// return success
	return _check();
}

//---------------------------------------------------------------------------------------
// Set Constraints
void CReconstructionAlgorithm2D::setConstraints(bool _bUseMin, float32 _fMinValue, bool _bUseMax, float32 _fMaxValue)
{
	m_bUseMinConstraint = _bUseMin;
	m_fMinValue = _fMinValue;
	m_bUseMaxConstraint = _bUseMax;
	m_fMaxValue = _fMaxValue;
}

//----------------------------------------------------------------------------------------
// Set Fixed Reconstruction Mask
void CReconstructionAlgorithm2D::setReconstructionMask(CFloat32VolumeData2D* _pMask, bool _bEnable)
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
void CReconstructionAlgorithm2D::setSinogramMask(CFloat32ProjectionData2D* _pMask, bool _bEnable)
{
	// TODO: check geometry matches sinogram
	m_bUseSinogramMask = _bEnable;
	m_pSinogramMask = _pMask;
	if (m_pSinogramMask == NULL) {
		m_bUseSinogramMask = false;
	}
}//----------------------------------------------------------------------------------------
// Check
bool CReconstructionAlgorithm2D::_check() 
{
	// check pointers
	if (requiresProjector())
		ASTRA_CONFIG_CHECK(m_pProjector, "Reconstruction2D", "Invalid Projector Object.");
	ASTRA_CONFIG_CHECK(m_pSinogram, "Reconstruction2D", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pReconstruction, "Reconstruction2D", "Invalid Reconstruction Data Object.");

	// check initializations
	if (requiresProjector())
		ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "Reconstruction2D", "Projector Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "Reconstruction2D", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isInitialized(), "Reconstruction2D", "Reconstruction Data Object Not Initialized.");

	// check compatibility between projector and data classes
	if (requiresProjector()) {
		ASTRA_CONFIG_CHECK(m_pSinogram->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "Reconstruction2D", "Projection Data not compatible with the specified Projector.");
		ASTRA_CONFIG_CHECK(m_pReconstruction->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "Reconstruction2D", "Reconstruction Data not compatible with the specified Projector.");
	}

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CReconstructionAlgorithm2D::getInformation() 
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
boost::any CReconstructionAlgorithm2D::getInformation(std::string _sIdentifier) 
{
	if (_sIdentifier == "UseMinConstraint")		{ return m_bUseMinConstraint ? string("yes") : string("no"); }
	if (_sIdentifier == "MinConstraintValue")	{ return m_fMinValue; }
	if (_sIdentifier == "UseMaxConstraint")		{ return m_bUseMaxConstraint ? string("yes") : string("no"); }
	if (_sIdentifier == "MaxConstraintValue")	{ return m_fMaxValue; }
	if (_sIdentifier == "ProjectorId")	{ 
		int iIndex = CProjector2DManager::getSingleton().getIndex(m_pProjector);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");	
	}
	if (_sIdentifier == "ProjectionDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pSinogram);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	} 
	if (_sIdentifier == "ReconstructionDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pReconstruction);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	if (_sIdentifier == "ReconstructionMaskId") {
		if (!m_bUseReconstructionMask) return string("not used");
		int iIndex = CData2DManager::getSingleton().getIndex(m_pReconstructionMask);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	return CAlgorithm::getInformation(_sIdentifier);
};
//----------------------------------------------------------------------------------------

} // namespace astra
