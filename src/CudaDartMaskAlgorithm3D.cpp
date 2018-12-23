/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#ifdef ASTRA_CUDA

#include "astra/CudaDartMaskAlgorithm3D.h"

#include "astra/cuda/3d/darthelper3d.h"
#include "astra/cuda/3d/dims3d.h"

#include "astra/AstraObjectManager.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaDartMaskAlgorithm3D::type = "DARTMASK3D_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaDartMaskAlgorithm3D::CCudaDartMaskAlgorithm3D() 
{
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaDartMaskAlgorithm3D::~CCudaDartMaskAlgorithm3D() 
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaDartMaskAlgorithm3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaDartMaskAlgorithm3D", this, _cfg);

	// reconstruction data
	XMLNode node = _cfg.self.getSingleNode("SegmentationDataId");
	ASTRA_CONFIG_CHECK(node, "CudaDartMask3D", "No SegmentationDataId tag specified.");
	int id = StringUtil::stringToInt(node.getContent(), -1);
	m_pSegmentation = dynamic_cast<CFloat32VolumeData3DMemory*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("SegmentationDataId");

	// reconstruction data
	node = _cfg.self.getSingleNode("MaskDataId");
	ASTRA_CONFIG_CHECK(node, "CudaDartMask3D", "No MaskDataId tag specified.");
	id = StringUtil::stringToInt(node.getContent(), -1);
	m_pMask = dynamic_cast<CFloat32VolumeData3DMemory*>(CData3DManager::getSingleton().get(id));
	CC.markNodeParsed("MaskDataId");

	// Option: GPU number
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUindex", -1);
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUIndex", m_iGPUIndex);
	CC.markOptionParsed("GPUindex");
	if (!_cfg.self.hasOption("GPUindex"))
		CC.markOptionParsed("GPUIndex");

	// Option: Connectivity
	try {
		m_iConn = _cfg.self.getOptionInt("Connectivity", 26);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaDartMask3D", "Connectivity must be an integer.");
	}
	CC.markOptionParsed("Connectivity");

	// Option: Threshold
	try {
		m_iThreshold = _cfg.self.getOptionInt("Threshold", 1);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaDartMask3D", "Threshold must be an integer.");
	}
	CC.markOptionParsed("Threshold");

	// Option: Radius
	try {
		m_iRadius = _cfg.self.getOptionInt("Radius", 1);
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "CudaDartMask3D", "Radius must be an integer.");
	}
	CC.markOptionParsed("Radius");

	_check();

	if (!m_bIsInitialized)
		return false;

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
//bool CCudaDartMaskAlgorithm3D::initialize(CFloat32VolumeData2D* _pSegmentation, int _iConn)
//{
//	return false;
//}

//----------------------------------------------------------------------------------------
// Iterate
void CCudaDartMaskAlgorithm3D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	const CVolumeGeometry3D& volgeom = *m_pSegmentation->getGeometry();
	astraCUDA3d::SDimensions3D dims;
	dims.iVolX = volgeom.getGridColCount();
	dims.iVolY = volgeom.getGridRowCount();
	dims.iVolZ = volgeom.getGridSliceCount();

	astraCUDA3d::setGPUIndex(m_iGPUIndex);
	astraCUDA3d::dartMasking(m_pMask->getData(), m_pSegmentation->getDataConst(), m_iConn, m_iRadius, m_iThreshold, dims);
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaDartMaskAlgorithm3D::_check() 
{

	// connectivity: 6 or 26
	ASTRA_CONFIG_CHECK(m_iConn == 6 || m_iConn == 26, "CudaDartMask3D", "Connectivity must be 6 or 26");

	// gpuindex >= 0 


	// success
	m_bIsInitialized = true;
	return true;
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CCudaDartMaskAlgorithm3D::getInformation()
{
	map<string,boost::any> res;
	// TODO: add PDART-specific options
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
}

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CCudaDartMaskAlgorithm3D::getInformation(std::string _sIdentifier)
{
	return NULL;
}


} // namespace astra

#endif // ASTRA_CUDA
