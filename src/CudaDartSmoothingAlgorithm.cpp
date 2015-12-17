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

#include "astra/CudaDartSmoothingAlgorithm.h"

#include "../cuda/2d/darthelper.h"
#include "../cuda/2d/algo.h"

#include "astra/AstraObjectManager.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaDartSmoothingAlgorithm::type = "DARTSMOOTHING_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaDartSmoothingAlgorithm::CCudaDartSmoothingAlgorithm() 
{
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaDartSmoothingAlgorithm::~CCudaDartSmoothingAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaDartSmoothingAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaDartSmoothingAlgorithm", this, _cfg);

	// reconstruction data
	XMLNode node = _cfg.self.getSingleNode("InDataId");
	ASTRA_CONFIG_CHECK(node, "CudaDartMask", "No InDataId tag specified.");
	int id = node.getContentInt();
	m_pIn = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("InDataId");

	// reconstruction data
	node = _cfg.self.getSingleNode("OutDataId");
	ASTRA_CONFIG_CHECK(node, "CudaDartMask", "No OutDataId tag specified.");
	id = node.getContentInt();
	m_pOut = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	CC.markNodeParsed("OutDataId");

	// Option: GPU number
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUindex", -1);
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUIndex", m_iGPUIndex);
	CC.markOptionParsed("GPUindex");
	if (!_cfg.self.hasOption("GPUindex"))
		CC.markOptionParsed("GPUIndex");

	// Option: Radius
	m_fB = (float)_cfg.self.getOptionNumerical("Intensity", 0.3f);
	CC.markOptionParsed("Intensity");

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
void CCudaDartSmoothingAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	const CVolumeGeometry2D& volgeom = *m_pIn->getGeometry();
	unsigned int width = volgeom.getGridColCount();
	unsigned int height = volgeom.getGridRowCount();

	astraCUDA::setGPUIndex(m_iGPUIndex);
	
	astraCUDA::dartSmoothing(m_pOut->getData(), m_pIn->getDataConst(), m_fB, m_iRadius, width, height);
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaDartSmoothingAlgorithm::_check() 
{
	// success
	m_bIsInitialized = true;
	return true;
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CCudaDartSmoothingAlgorithm::getInformation()
{
	map<string,boost::any> res;
	// TODO: add PDART-specific options
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
}

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CCudaDartSmoothingAlgorithm::getInformation(std::string _sIdentifier)
{
	return NULL;
}


} // namespace astra

#endif // ASTRA_CUDA
