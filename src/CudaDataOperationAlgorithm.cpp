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

#include "astra/CudaDataOperationAlgorithm.h"

#include "../cuda/2d/algo.h"
#include "../cuda/2d/darthelper.h"
#include "../cuda/2d/arith.h"

#include "astra/AstraObjectManager.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaDataOperationAlgorithm::type = "DataOperation_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaDataOperationAlgorithm::CCudaDataOperationAlgorithm() 
{
	m_pMask = NULL;
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaDataOperationAlgorithm::~CCudaDataOperationAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaDataOperationAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CCudaDataOperationAlgorithm", this, _cfg);

	// operation
	XMLNode node = _cfg.self.getSingleNode("Operation");
	ASTRA_CONFIG_CHECK(node, "CCudaDataOperationAlgorithm", "No Operation tag specified.");
	m_sOperation = node.getContent();
	m_sOperation.erase(std::remove(m_sOperation.begin(), m_sOperation.end(), ' '), m_sOperation.end());
	CC.markNodeParsed("Operation");

	// data
	node = _cfg.self.getSingleNode("DataId");
	ASTRA_CONFIG_CHECK(node, "CCudaDataOperationAlgorithm", "No DataId tag specified.");
	vector<string> data = node.getContentArray();
	for (vector<string>::iterator it = data.begin(); it != data.end(); ++it){
		int id = StringUtil::stringToInt(*it);
		m_pData.push_back(dynamic_cast<CFloat32Data2D*>(CData2DManager::getSingleton().get(id)));
	}
	CC.markNodeParsed("DataId");

	// scalar
	node = _cfg.self.getSingleNode("Scalar");
	ASTRA_CONFIG_CHECK(node, "CCudaDataOperationAlgorithm", "No Scalar tag specified.");
	m_fScalar = node.getContentNumericalArray();
	CC.markNodeParsed("Scalar");

	// Option: GPU number
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUindex", -1);
	m_iGPUIndex = (int)_cfg.self.getOptionNumerical("GPUIndex", m_iGPUIndex);
	CC.markOptionParsed("GPUindex");
	if (!_cfg.self.hasOption("GPUindex"))
		CC.markOptionParsed("GPUIndex");

	if (_cfg.self.hasOption("MaskId")) {
		int id = _cfg.self.getOptionInt("MaskId");
		m_pMask = dynamic_cast<CFloat32Data2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("MaskId");

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
void CCudaDataOperationAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	astraCUDA::setGPUIndex(m_iGPUIndex);

	astraCUDA::SDimensions dims;
	// We slightly abuse dims here: width/height is not necessarily a volume
	dims.iVolWidth = m_pData[0]->getWidth();
	dims.iVolHeight = m_pData[0]->getHeight();

	if (m_sOperation == "$1*s1" || m_sOperation == "$1.*s1") // data * scalar
	{
		if (m_pMask == NULL)
			astraCUDA::processVolCopy<astraCUDA::opMul>(m_pData[0]->getData(), m_fScalar[0], dims);
		else
			astraCUDA::processVolCopy<astraCUDA::opMulMask>(m_pData[0]->getData(), m_pMask->getDataConst(), m_fScalar[0], dims);
	}
	else if (m_sOperation == "$1/s1" || m_sOperation == "$1./s1") // data / scalar
	{
		if (m_pMask == NULL)
			astraCUDA::processVolCopy<astraCUDA::opMul>(m_pData[0]->getData(), 1.0f/m_fScalar[0], dims);
		else
			astraCUDA::processVolCopy<astraCUDA::opMulMask>(m_pData[0]->getData(), m_pMask->getDataConst(), 1.0f/m_fScalar[0], dims);
	}
	else if (m_sOperation == "$1+s1") // data + scalar
	{
		astraCUDA::processVolCopy<astraCUDA::opAdd>(m_pData[0]->getData(), m_fScalar[0], dims);
	}
	else if (m_sOperation == "$1-s1") // data - scalar
	{
		astraCUDA::processVolCopy<astraCUDA::opAdd>(m_pData[0]->getData(), -m_fScalar[0], dims);
	} 
	else if (m_sOperation == "$1.*$2") // data .* data
	{
		astraCUDA::processVolCopy<astraCUDA::opMul>(m_pData[0]->getData(), m_pData[1]->getDataConst(), dims);
	}
	else if (m_sOperation == "$1+$2") // data + data
	{
		astraCUDA::processVolCopy<astraCUDA::opAdd>(m_pData[0]->getData(), m_pData[1]->getDataConst(), dims);
	}

}

//----------------------------------------------------------------------------------------
// Check
bool CCudaDataOperationAlgorithm::_check() 
{
	// s*: 1 data + 1 scalar

	// success
	m_bIsInitialized = true;
	return true;
}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CCudaDataOperationAlgorithm::getInformation()
{
	map<string,boost::any> res;
	// TODO: add PDART-specific options
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), res);
}

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CCudaDataOperationAlgorithm::getInformation(std::string _sIdentifier)
{
	return NULL;
}


} // namespace astra

#endif // ASTRA_CUDA
