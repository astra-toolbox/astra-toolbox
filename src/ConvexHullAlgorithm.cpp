/*
-----------------------------------------------------------------------
Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
                2014, CWI, Amsterdam

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

#include "astra/ConvexHullAlgorithm.h"

#include <boost/lexical_cast.hpp>

#include "astra/AstraObjectManager.h"
#include "astra/DataProjectorPolicies.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CConvexHullAlgorithm::type = "ConvexHull";

//----------------------------------------------------------------------------------------
// Constructor
CConvexHullAlgorithm::CConvexHullAlgorithm() 
{
	_clear();
}

//---------------------------------------------------------------------------------------
// Initialize - C++
CConvexHullAlgorithm::CConvexHullAlgorithm(CProjector2D* _pProjector, 
										   CFloat32ProjectionData2D* _pSinogram, 
										   CFloat32VolumeData2D* _pReconstructionMask)
{
	_clear();
	initialize(_pProjector, _pSinogram, _pReconstructionMask);
}

//----------------------------------------------------------------------------------------
// Destructor
CConvexHullAlgorithm::~CConvexHullAlgorithm() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CConvexHullAlgorithm::_clear()
{
	m_bIsInitialized = false;

	m_pProjectionPixelWeight = NULL;
	m_pReconstructionMask = NULL;
	m_pSinogram = NULL;

	m_pProjector = NULL;
	m_pDataProjector = NULL;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CConvexHullAlgorithm::clear()
{
	m_bIsInitialized = false;

	ASTRA_DELETE(m_pProjectionPixelWeight);
	m_pReconstructionMask = NULL;
	m_pSinogram = NULL;

	m_pProjector = NULL;
	ASTRA_DELETE(m_pDataProjector);
}

//----------------------------------------------------------------------------------------
// Check
bool CConvexHullAlgorithm::_check()
{
	ASTRA_CONFIG_CHECK(m_pReconstructionMask, "ConvexHull", "Invalid ReconstructionMask Object");
	ASTRA_CONFIG_CHECK(m_pReconstructionMask->isInitialized(), "ConvexHull", "Invalid ReconstructionMask Object");
	ASTRA_CONFIG_CHECK(m_pProjectionPixelWeight, "ConvexHull", "Invalid ProjectionPixelWeight Object");
	ASTRA_CONFIG_CHECK(m_pProjectionPixelWeight->isInitialized(), "ConvexHull", "Invalid ProjectionPixelWeight Object");
	ASTRA_CONFIG_CHECK(m_pSinogram, "ConvexHull", "Invalid Sinogram Object");
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "ConvexHull", "Invalid Sinogram Object");

	ASTRA_CONFIG_CHECK(m_pDataProjector, "ConvexHull", "Invalid Data Projector Policy");
	ASTRA_CONFIG_CHECK(m_pProjector, "ConvexHull", "Invalid Projector Object");
	ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "ConvexHull", "Invalid Projector Object");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CConvexHullAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// projector
	XMLNode* node = _cfg.self->getSingleNode("ProjectorId");
	ASTRA_CONFIG_CHECK(node, "ConvexHull", "No ProjectorId tag specified.");
	int id = boost::lexical_cast<int>(node->getContent());
	m_pProjector = CProjector2DManager::getSingleton().get(id);
	ASTRA_DELETE(node);

	// sinogram data
	node = _cfg.self->getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "ConvexHull", "No ProjectionDataId tag specified.");
	id = boost::lexical_cast<int>(node->getContent());
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	ASTRA_DELETE(node);

	// reconstruction mask
	node = _cfg.self->getSingleNode("ConvexHullDataId");
	ASTRA_CONFIG_CHECK(node, "ConvexHull", "No ReconstructionDataId tag specified.");
	id = boost::lexical_cast<int>(node->getContent());
	m_pReconstructionMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	ASTRA_DELETE(node);

	// init data objects and data projectors
	_init();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CConvexHullAlgorithm::initialize(CProjector2D* _pProjector, 
									  CFloat32ProjectionData2D* _pSinogram, 
									  CFloat32VolumeData2D* _pReconstructionMask)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstructionMask = _pReconstructionMask;

	// init data objects and data projectors
	_init();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize Data Projectors - private
void CConvexHullAlgorithm::_init()
{
	// create data objects
	m_pProjectionPixelWeight = new CFloat32VolumeData2D(m_pProjector->getVolumeGeometry());
	m_pProjectionPixelWeight->setData(0);

	// forward projection data projector
	m_pDataProjector = dispatchDataProjector(
		m_pProjector, 
			//SinogramMaskPolicy(m_pSinogramMask),										// sinogram mask
			TotalPixelWeightBySinogramPolicy(m_pSinogram, m_pProjectionPixelWeight)		// pixel weight * sinogram		
		);
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CConvexHullAlgorithm::getInformation() 
{
	map<string, boost::any> res;
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
};

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CConvexHullAlgorithm::getInformation(std::string _sIdentifier) 
{
	return CAlgorithm::getInformation(_sIdentifier);
};

//----------------------------------------------------------------------------------------
// Iterate
void CConvexHullAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	m_pReconstructionMask->setData(1.0f);

	// loop angles
	for (int iProjection = 0; iProjection < m_pProjector->getProjectionGeometry()->getProjectionAngleCount(); ++iProjection) {
		
		m_pProjectionPixelWeight->setData(0.0f);

		// project
		m_pDataProjector->projectSingleProjection(iProjection);

		// loop values and set to zero
		for (int iPixel = 0; iPixel < m_pReconstructionMask->getSize(); ++iPixel) {
			if (m_pProjectionPixelWeight->getData()[iPixel] == 0) {
				m_pReconstructionMask->getData()[iPixel] = 0;
			}
		}

	}

}
//----------------------------------------------------------------------------------------

} // namespace astra
