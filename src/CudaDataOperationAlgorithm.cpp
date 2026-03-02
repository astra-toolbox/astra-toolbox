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
#include "astra/cuda/2d/arith.h"
#include "astra/cuda/2d/mem2d.h"

#include "astra/AstraObjectManager.h"
#include "astra/Data2D.h"

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
		m_pData.push_back(dynamic_cast<CData2D*>(CData2DManager::getSingleton().get(*it)));
	}

	bool ok = true;

	ok &= CR.getRequiredNumericalArray("Scalar", m_fScalar);

	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);

	int id = -1;
	if (CR.getOptionID("MaskId", id)) {
		m_pMask = dynamic_cast<CData2D*>(CData2DManager::getSingleton().get(id));
	}

	if (!ok)
		return false;

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

	bool ok = true;

	std::vector<CData2D*> D_data;
	for (CData2D *data : m_pData) {
		std::array<int, 2> volDims = data->getShape();
		CDataStorage *s = astraCUDA::allocateGPUMemory(volDims[0], volDims[1], astraCUDA::INIT_NO);
		if (!s) {
			ok = false;
			break;
		}
		CData2D *d = new CData2D(volDims[0], volDims[1], s);
		ok &= astraCUDA::copyToGPUMemory(data, d);
		if (!ok)
			break;
		D_data.push_back(d);
	}
	CData2D *D_mask = nullptr;
	if (ok && m_pMask) {
		std::array<int, 2> volDims = m_pMask->getShape();
		CDataStorage *s = astraCUDA::allocateGPUMemory(volDims[0], volDims[1], astraCUDA::INIT_NO);
		if (s) {
			D_mask = new CData2D(volDims[0], volDims[1], s);
			ok &= astraCUDA::copyToGPUMemory(m_pMask, D_mask);
		} else {
			ok = false;
		}
	}

	if (ok) {
		if (m_sOperation == "$1*s1" || m_sOperation == "$1.*s1") // data * scalar
		{
			if (m_pMask == NULL)
				ok &= astraCUDA::processData<astraCUDA::opMul>(D_data[0], m_fScalar[0]);
			else
				ok &= astraCUDA::processData<astraCUDA::opMulMask>(D_data[0], D_mask, m_fScalar[0]);
		}
		else if (m_sOperation == "$1/s1" || m_sOperation == "$1./s1") // data / scalar
		{
			if (m_pMask == NULL)
				ok &= astraCUDA::processData<astraCUDA::opMul>(D_data[0], 1.0f/m_fScalar[0]);
			else
				ok &= astraCUDA::processData<astraCUDA::opMulMask>(D_data[0], D_mask, 1.0f/m_fScalar[0]);
		}
		else if (m_sOperation == "$1+s1") // data + scalar
		{
			ok &= astraCUDA::processData<astraCUDA::opAdd>(D_data[0], m_fScalar[0]);
		}
		else if (m_sOperation == "$1-s1") // data - scalar
		{
			ok &= astraCUDA::processData<astraCUDA::opAdd>(D_data[0], -m_fScalar[0]);
		}
		else if (m_sOperation == "$1.*$2") // data .* data
		{
			ok &= astraCUDA::processData<astraCUDA::opMul>(D_data[0], D_data[1]);
		}
		else if (m_sOperation == "$1+$2") // data + data
		{
			ok &= astraCUDA::processData<astraCUDA::opAdd>(D_data[0], D_data[1]);
		}
		else
			ok = false;
	}

	// Only transfer the first data object back to host memory
	ok &= astraCUDA::copyFromGPUMemory(m_pData[0], D_data[0]);

	for (CData2D *d : D_data) {
		astraCUDA::freeGPUMemory(d);
		delete d;
	}
	if (D_mask) {
		astraCUDA::freeGPUMemory(D_mask);
		delete D_mask;
	}

	return ok;
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
