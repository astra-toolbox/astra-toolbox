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

#include "astra/ForwardProjectionAlgorithm.h"

#include "astra/AstraObjectManager.h"
#include "astra/DataProjectorPolicies.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CForwardProjectionAlgorithm::type = "FP";

//----------------------------------------------------------------------------------------
// Constructor - Default
CForwardProjectionAlgorithm::CForwardProjectionAlgorithm() 
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Constructor
CForwardProjectionAlgorithm::CForwardProjectionAlgorithm(CProjector2D* _pProjector, CFloat32VolumeData2D* _pVolume, CFloat32ProjectionData2D* _pSinogram)
{
	_clear();
	initialize(_pProjector, _pVolume, _pSinogram);
}

//----------------------------------------------------------------------------------------
// Destructor
CForwardProjectionAlgorithm::~CForwardProjectionAlgorithm() 
{
	delete m_pForwardProjector;
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CForwardProjectionAlgorithm::_clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pVolume = NULL;
	m_pForwardProjector = NULL;
	m_bUseSinogramMask = false;
	m_bUseVolumeMask = false;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CForwardProjectionAlgorithm::clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pVolume = NULL;
	m_bUseSinogramMask = false;
	m_bUseVolumeMask = false;
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check
bool CForwardProjectionAlgorithm::_check() 
{
	// check pointers
	ASTRA_CONFIG_CHECK(m_pProjector, "ForwardProjection", "Invalid Projector Object.");
	ASTRA_CONFIG_CHECK(m_pSinogram, "ForwardProjection", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pVolume, "ForwardProjection", "Invalid Volume Data Object.");

	// check initializations
	ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "ForwardProjection", "Projector Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "ForwardProjection", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pVolume->isInitialized(), "ForwardProjection", "Volume Data Object Not Initialized.");

	// check compatibility between projector and data classes
	ASTRA_CONFIG_CHECK(m_pSinogram->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "ForwardProjection", "Projection Data not compatible with the specified Projector.");
	ASTRA_CONFIG_CHECK(m_pVolume->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "ForwardProjection", "Volume Data not compatible with the specified Projector.");

	ASTRA_CONFIG_CHECK(m_pForwardProjector, "ForwardProjection", "Invalid FP Policy");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CForwardProjectionAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// projector
	XMLNode node = _cfg.self.getSingleNode("ProjectorId");
	ASTRA_CONFIG_CHECK(node, "ForwardProjection", "No ProjectorId tag specified.");
	int id = node.getContentInt();
	m_pProjector = CProjector2DManager::getSingleton().get(id);

	// sinogram data
	node = _cfg.self.getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "ForwardProjection", "No ProjectionDataId tag specified.");
	id = node.getContentInt();
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));

	// volume data
	node = _cfg.self.getSingleNode("VolumeDataId");
	ASTRA_CONFIG_CHECK(node, "ForwardProjection", "No VolumeDataId tag specified.");
	id = node.getContentInt();
	m_pVolume = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	
	// volume mask
	if (_cfg.self.hasOption("VolumeMaskId")) {
		m_bUseVolumeMask = true;
		id = _cfg.self.getOptionInt("VolumeMaskId");
		m_pVolumeMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}

	// sino mask
	if (_cfg.self.hasOption("SinogramMaskId")) {
		m_bUseSinogramMask = true;
		id = _cfg.self.getOptionInt("SinogramMaskId");
		m_pSinogramMask = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	}

	// ray or voxel-driven projector?
	//m_bUseVoxelProjector = _cfg.self->getOptionBool("VoxelDriven", false);

	// init data projector
	_init();

	// return success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Get Information - all
map<string,boost::any> CForwardProjectionAlgorithm::getInformation() 
{
	map<string, boost::any> result;
	result["ProjectorId"] = getInformation("ProjectorId");
	result["ProjectionDataId"] = getInformation("ProjectionDataId");
	result["VolumeDataId"] = getInformation("VolumeDataId");
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), result);
};

//---------------------------------------------------------------------------------------
// Get Information - specific
boost::any CForwardProjectionAlgorithm::getInformation(std::string _sIdentifier) 
{
	if (_sIdentifier == "ProjectorId") {
		int iIndex = CProjector2DManager::getSingleton().getIndex(m_pProjector);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	} else if (_sIdentifier == "ProjectionDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pSinogram);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	} else if (_sIdentifier == "VolumeDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pVolume);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	return CAlgorithm::getInformation(_sIdentifier);
};

//----------------------------------------------------------------------------------------
// Initialize
bool CForwardProjectionAlgorithm::initialize(CProjector2D* _pProjector, 
											 CFloat32VolumeData2D* _pVolume,
											 CFloat32ProjectionData2D* _pSinogram)
{
	// store classes
	m_pProjector = _pProjector;
	m_pVolume = _pVolume;
	m_pSinogram = _pSinogram;

	// init data projector
	_init();

	// return success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize Data Projectors - private
void CForwardProjectionAlgorithm::_init()
{
	// forward projection data projector
	m_pForwardProjector = dispatchDataProjector(
		m_pProjector, 
		SinogramMaskPolicy(m_pSinogramMask),			// sinogram mask
		ReconstructionMaskPolicy(m_pVolumeMask),		// reconstruction mask
		DefaultFPPolicy(m_pVolume, m_pSinogram),		// forward projection
		m_bUseSinogramMask, m_bUseVolumeMask, true		// options on/off
	); 
}

//----------------------------------------------------------------------------------------
// Set Fixed Reconstruction Mask
void CForwardProjectionAlgorithm::setVolumeMask(CFloat32VolumeData2D* _pMask, bool _bEnable)
{
	// TODO: check geometry matches volume
	m_bUseVolumeMask = _bEnable;
	m_pVolumeMask = _pMask;
	if (m_pVolumeMask == NULL) {
		m_bUseVolumeMask = false;
	}
}

//----------------------------------------------------------------------------------------
// Set Fixed Sinogram Mask
void CForwardProjectionAlgorithm::setSinogramMask(CFloat32ProjectionData2D* _pMask, bool _bEnable)
{
	// TODO: check geometry matches sinogram
	m_bUseSinogramMask = _bEnable;
	m_pSinogramMask = _pMask;
	if (m_pSinogramMask == NULL) {
		m_bUseSinogramMask = false;
	}
}

//----------------------------------------------------------------------------------------
// Iterate
void CForwardProjectionAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	m_pSinogram->setData(0.0f);

//	if (m_bUseVoxelProjector) {
//		m_pForwardProjector->projectAllVoxels();
//	} else {
		m_pForwardProjector->project();
//	}

}
//----------------------------------------------------------------------------------------

} // namespace astra
