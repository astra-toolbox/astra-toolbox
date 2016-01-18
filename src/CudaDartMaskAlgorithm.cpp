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

#ifdef ASTRA_CUDA

#include "astra/CudaDartMaskAlgorithm.h"

#include "../cuda/2d/darthelper.h"
#include "../cuda/2d/algo.h"

#include "astra/AstraObjectManager.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaDartMaskAlgorithm::type = "DARTMASK_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaDartMaskAlgorithm::CCudaDartMaskAlgorithm() 
{
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaDartMaskAlgorithm::~CCudaDartMaskAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaDartMaskAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaDartMaskAlgorithm", this, _cfg);

	// reconstruction data
	XMLNode node = _cfg.self.getSingleNode("SegmentationDataId");
	ASTRA_CONFIG_CHECK(node, "CudaDartMask", "No SegmentationDataId tag specified.");
	int id = node.getContentInt();
	m_pSegmentation = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("SegmentationDataId");

	// reconstruction data
	node = _cfg.self.getSingleNode("MaskDataId");
	ASTRA_CONFIG_CHECK(node, "CudaDartMask", "No MaskDataId tag specified.");
	id = node.getContentInt();
	m_pMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("MaskDataId");

	// Option: GPU number
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUindex", -1);
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUIndex", m_iGPUIndex);
	CC.markOptionParsed("GPUindex");
	if (!_cfg.self.hasOption("GPUindex"))
		CC.markOptionParsed("GPUIndex");

    // Option: Connectivity
	m_iConn = (unsigned int)_cfg.self.getOptionNumerical("Connectivity", 8);
	CC.markOptionParsed("Connectivity");

	// Option: Threshold
	m_iThreshold = (unsigned int)_cfg.self.getOptionNumerical("Threshold", 1);
	CC.markOptionParsed("Threshold");

	// Option: Radius
	m_iRadius = (unsigned int)_cfg.self.getOptionNumerical("Radius", 1);
	CC.markOptionParsed("Radius");

	_check();

	if (!m_bIsInitialized)
		return false;

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
//bool CCudaDartMaskAlgorithm::initialize(CFloat32VolumeData2D* _pSegmentation, int _iConn)
//{
//	return false;
//}

//----------------------------------------------------------------------------------------
// Iterate
void CCudaDartMaskAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	const CVolumeGeometry2D& volgeom = *m_pSegmentation->getGeometry();
	unsigned int width = volgeom.getGridColCount();
	unsigned int height = volgeom.getGridRowCount();

	astraCUDA::setGPUIndex(m_iGPUIndex);
	astraCUDA::dartMask(m_pMask->getData(), m_pSegmentation->getDataConst(), m_iConn, m_iRadius, m_iThreshold, width, height);
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaDartMaskAlgorithm::_check() 
{

	// connectivity: 4 of 8

	// gpuindex >= 0 


	// success
	m_bIsInitialized = true;
	return true;
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CCudaDartMaskAlgorithm::getInformation()
{
	map<string,boost::any> res;
	// TODO: add PDART-specific options
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
}

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CCudaDartMaskAlgorithm::getInformation(std::string _sIdentifier)
{
	return NULL;
}


} // namespace astra

#endif // ASTRA_CUDA
