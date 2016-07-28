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

#include "astra/ReconstructionAlgorithmMultiSlice2D.h"

#include "astra/AstraObjectManager.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CReconstructionAlgorithmMultiSlice2D::CReconstructionAlgorithmMultiSlice2D() 
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CReconstructionAlgorithmMultiSlice2D::~CReconstructionAlgorithmMultiSlice2D() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CReconstructionAlgorithmMultiSlice2D::_clear()
{
	m_pProjector = NULL;
	m_iSliceCount = 0;
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
void CReconstructionAlgorithmMultiSlice2D::clear()
{
	m_pProjector = NULL;
	m_vpSinogram.clear();
	m_vpReconstruction.clear();
	m_iSliceCount = 0;
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
bool CReconstructionAlgorithmMultiSlice2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("ReconstructionAlgorithmMultiSlice2D", this, _cfg);
	
	// projector
	XMLNode* node = _cfg.self->getSingleNode("ProjectorId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction2D", "No ProjectorId tag specified.");
	int id = node->getContentInt();
	m_pProjector = CProjector2DManager::getSingleton().get(id);
	ASTRA_DELETE(node);
	CC.markNodeParsed("ProjectorId");

	// sinogram data
	node = _cfg.self->getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction2D", "No ProjectionDataId tag specified.");
	vector<float32> tmpvector = node->getContentNumericalArray();
	for (unsigned int i = 0; i < tmpvector.size(); ++i) {
		m_vpSinogram.push_back(dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(int(tmpvector[i]))));
	}
	m_iSliceCount = tmpvector.size();
	ASTRA_DELETE(node);
	CC.markNodeParsed("ProjectionDataId");

	// reconstruction data
	node = _cfg.self->getSingleNode("ReconstructionDataId");
	ASTRA_CONFIG_CHECK(node, "Reconstruction2D", "No ReconstructionDataId tag specified.");
	tmpvector =  node->getContentNumericalArray();
	for (unsigned int i = 0; i < tmpvector.size(); ++i) {
		m_vpReconstruction.push_back(dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(int(tmpvector[i]))));
	}
	ASTRA_DELETE(node);
	CC.markNodeParsed("ReconstructionDataId");

	// reconstruction masks
	if (_cfg.self->hasOption("ReconstructionMaskId")) {
		m_bUseReconstructionMask = true;
		id = _cfg.self->getOptionInt("ReconstructionMaskId");
		m_pReconstructionMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("ReconstructionMaskId");

	// sinogram masks
	if (_cfg.self->hasOption("SinogramMaskId")) {
		m_bUseSinogramMask = true;
		id = _cfg.self->getOptionInt("SinogramMaskId");
		m_pSinogramMask = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("SinogramMaskId");

	// Constraints - NEW
	if (_cfg.self->hasOption("MinConstraint")) {
		m_bUseMinConstraint = true;
		m_fMinValue = _cfg.self->getOptionNumerical("MinConstraint", 0.0f);
		CC.markOptionParsed("MinConstraint");
	} else {
		// Constraint - OLD
		m_bUseMinConstraint = _cfg.self->getOptionBool("UseMinConstraint", false);
		CC.markOptionParsed("UseMinConstraint");
		if (m_bUseMinConstraint) {
			m_fMinValue = _cfg.self->getOptionNumerical("MinConstraintValue", 0.0f);
			CC.markOptionParsed("MinConstraintValue");
		}
	}
	if (_cfg.self->hasOption("MaxConstraint")) {
		m_bUseMaxConstraint = true;
		m_fMaxValue = _cfg.self->getOptionNumerical("MaxConstraint", 255.0f);
		CC.markOptionParsed("MaxConstraint");
	} else {
		// Constraint - OLD
		m_bUseMaxConstraint = _cfg.self->getOptionBool("UseMaxConstraint", false);
		CC.markOptionParsed("UseMaxConstraint");
		if (m_bUseMaxConstraint) {
			m_fMaxValue = _cfg.self->getOptionNumerical("MaxConstraintValue", 0.0f);
			CC.markOptionParsed("MaxConstraintValue");
		}
	}

	// return success
	return _check();
}

//----------------------------------------------------------------------------------------
// Initialize - C++
bool CReconstructionAlgorithmMultiSlice2D::initialize(CProjector2D* _pProjector, 
													  vector<CFloat32ProjectionData2D*> _vpSinogram, 
													  vector<CFloat32VolumeData2D*> _vpReconstruction)
{
	m_pProjector = _pProjector;
	m_vpSinogram = _vpSinogram;
	m_vpReconstruction = _vpReconstruction;
	m_iSliceCount = _vpSinogram.size();

	// return success
	return _check();
}

//---------------------------------------------------------------------------------------
// Set Constraints
void CReconstructionAlgorithmMultiSlice2D::setConstraints(bool _bUseMin, float32 _fMinValue, bool _bUseMax, float32 _fMaxValue)
{
	m_bUseMinConstraint = _bUseMin;
	m_fMinValue = _fMinValue;
	m_bUseMaxConstraint = _bUseMax;
	m_fMaxValue = _fMaxValue;
}

//----------------------------------------------------------------------------------------
// Set Fixed Reconstruction Mask
void CReconstructionAlgorithmMultiSlice2D::setReconstructionMask(CFloat32VolumeData2D* _pMask, bool _bEnable)
{
	m_bUseReconstructionMask = _bEnable;
	m_pReconstructionMask = _pMask;
	if (m_pReconstructionMask == NULL) {
		m_bUseReconstructionMask = false;
	}
}

//----------------------------------------------------------------------------------------
// Check
bool CReconstructionAlgorithmMultiSlice2D::_check() 
{
	// check projector
	ASTRA_CONFIG_CHECK(m_pProjector, "ReconstructionMultiSlice2D", "Invalid Projector Object.");
	ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "ReconstructionMultiSlice2D", "Projector Object Not Initialized.");

	// check list
	ASTRA_CONFIG_CHECK(m_vpSinogram.size() == (unsigned int)m_iSliceCount, "ReconstructionMultiSlice2D", "Sinogram slicecount mismatch.");
	ASTRA_CONFIG_CHECK(m_vpReconstruction.size() == (unsigned int)m_iSliceCount, "ReconstructionMultiSlice2D", "Volume slicecount mismatch.");

	for (int i = 0; i < m_iSliceCount; ++i) {
		// pointers
		ASTRA_CONFIG_CHECK(m_vpSinogram[i], "ReconstructionMultiSlice2D", "Invalid Projection Data Object.");
		ASTRA_CONFIG_CHECK(m_vpReconstruction[i], "ReconstructionMultiSlice2D", "Invalid Volume Data Object.");

		// initialized
		ASTRA_CONFIG_CHECK(m_vpSinogram[i]->isInitialized(), "ReconstructionMultiSlice2D", "Projection Data Object Not Initialized.");
		ASTRA_CONFIG_CHECK(m_vpReconstruction[i]->isInitialized(), "ReconstructionMultiSlice2D", "Volume Data Object Not Initialized.");

		// geometry compatibility
		ASTRA_CONFIG_CHECK(m_vpSinogram[i]->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "Reconstruction2D", "Projection Data not compatible with the specified Projector.");
		ASTRA_CONFIG_CHECK(m_vpReconstruction[i]->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "Reconstruction2D", "Reconstruction Data not compatible with the specified Projector.");
	}

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CReconstructionAlgorithmMultiSlice2D::getInformation() 
{
	map<string, boost::any> res;
	res["ProjectorId"] = getInformation("ProjectorId");
//	res["ProjectionDataId"] = getInformation("ProjectionDataId");
//	res["ReconstructionDataId"] = getInformation("ReconstructionDataId");
	res["UseMinConstraint"] = getInformation("UseMinConstraint");
	res["MinConstraintValue"] = getInformation("MinConstraintValue");
	res["UseMaxConstraint"] = getInformation("UseMaxConstraint");
	res["MaxConstraintValue"] = getInformation("MaxConstraintValue");
	res["ReconstructionMaskId"] = getInformation("ReconstructionMaskId");
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
};

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CReconstructionAlgorithmMultiSlice2D::getInformation(std::string _sIdentifier) 
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
//	if (_sIdentifier == "ProjectionDataId") {
//		int iIndex = CData2DManager::getSingleton().getIndex(m_pSinogram);
//		if (iIndex != 0) return iIndex;
//		return std::string("not in manager");
//	} 
//	if (_sIdentifier == "ReconstructionDataId") {
//		int iIndex = CData2DManager::getSingleton().getIndex(m_pReconstruction);
//		if (iIndex != 0) return iIndex;
//		return std::string("not in manager");
//	}
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
