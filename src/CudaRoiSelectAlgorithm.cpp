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

#include "astra/CudaRoiSelectAlgorithm.h"

#include "astra/cuda/2d/astra.h"
#include "astra/cuda/2d/darthelper.h"
#include "astra/cuda/2d/algo.h"

#include "astra/AstraObjectManager.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaRoiSelectAlgorithm::CCudaRoiSelectAlgorithm() 
{
	m_fRadius = 0.0f;
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaRoiSelectAlgorithm::~CCudaRoiSelectAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaRoiSelectAlgorithm::initialize(const Config& _cfg)
{
	ConfigReader<CAlgorithm> CR("CudaDartRoiSelectAlgorithm", this, _cfg);

	bool ok = true;
	int id = -1;

	ok &= CR.getRequiredID("DataId", id);
	m_pData = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));

	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);

	ok &= CR.getOptionNumerical("Radius", m_fRadius, 0.0f);

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
bool CCudaRoiSelectAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	const CVolumeGeometry2D& volgeom = m_pData->getGeometry();
	unsigned int width = volgeom.getGridColCount();
	unsigned int height = volgeom.getGridRowCount();

	if (m_fRadius == 0){
		m_fRadius = (width < height) ? width : height;
	}

	astraCUDA::setGPUIndex(m_iGPUIndex);
	astraCUDA::roiSelect(m_pData->getData(), m_fRadius, width, height);

	return true;
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaRoiSelectAlgorithm::_check() 
{

	// success
	m_bIsInitialized = true;
	return true;
}

} // namespace astra

#endif // ASTRA_CUDA
