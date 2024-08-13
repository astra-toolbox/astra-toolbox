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

#ifdef ASTRA_CUDA

#include "astra/CudaDataOperationAlgorithm.h"

#include "astra/cuda/2d/astra.h"

#include "astra/AstraObjectManager.h"

#include "astra/Logging.h"

#include <algorithm>

using namespace std;

namespace astra {

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
	ConfigReader<CAlgorithm> CR("CCudaDataOperationAlgorithm", this, _cfg);

	// operation
	if (!CR.getRequiredString("Operation", m_sOperation))
		return false;
	m_sOperation.erase(std::remove(m_sOperation.begin(), m_sOperation.end(), ' '), m_sOperation.end());

	// data
	vector<int> data;
	if (!CR.getRequiredIntArray("DataId", data))
		return false;
	for (vector<int>::iterator it = data.begin(); it != data.end(); ++it){
		m_pData.push_back(dynamic_cast<CFloat32Data2D*>(CData2DManager::getSingleton().get(*it)));
	}

	bool ok = true;

	ok &= CR.getRequiredNumericalArray("Scalar", m_fScalar);

	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);

	int id = -1;
	if (CR.getOptionID("MaskId", id)) {
		m_pMask = dynamic_cast<CFloat32Data2D*>(CData2DManager::getSingleton().get(id));
	}

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
bool CCudaDataOperationAlgorithm::run(int _iNrIterations)
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

	return true;
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

} // namespace astra

#endif // ASTRA_CUDA
