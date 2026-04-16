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

#include "astra/CudaBackProjectionAlgorithm.h"

#include "astra/cuda/2d/astra.h"
#include "astra/cuda/2d/mem2d.h"

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaBackProjectionAlgorithm::CCudaBackProjectionAlgorithm() 
{

}

//----------------------------------------------------------------------------------------
// Destructor
CCudaBackProjectionAlgorithm::~CCudaBackProjectionAlgorithm() 
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaBackProjectionAlgorithm::initialize(const Config& _cfg)
{
	assert(!m_bIsInitialized);

	ConfigReader<CAlgorithm> CR("CudaBackProjectionAlgorithm", this, _cfg);

	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_cfg);

	if (!m_bIsInitialized)
		return false;

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaBackProjectionAlgorithm::initialize(CProjector2D* _pProjector,
                                              CFloat32ProjectionData2D* _pSinogram, 
                                              CFloat32VolumeData2D* _pReconstruction)
{
	assert(!m_bIsInitialized);

	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction);

	if (!m_bIsInitialized)
		return false;

	return true;
}

bool CCudaBackProjectionAlgorithm::run(int /*_iNrIterations*/)
{
	assert(m_bIsInitialized);

	bool ok;

	std::array<int, 2> volDims = m_pReconstruction->getShape();
	std::array<int, 2> projDims = m_pSinogram->getShape();

	if (m_iGPUIndex != -1)
		astraCUDA::setGPUIndex(m_iGPUIndex);

	CDataStorage *s;
	s = astraCUDA::allocateGPUMemory(volDims[0], volDims[1], astraCUDA::INIT_ZERO);
	if (!s) {
		return false;
	}
	CData2D *D_volData = new CData2D(volDims[0], volDims[1], s);

	s = astraCUDA::allocateGPUMemory(projDims[0], projDims[1], astraCUDA::INIT_NO);
	if (!s) {
		astraCUDA::freeGPUMemory(D_volData);
		delete D_volData;
		return false;
	}
	CData2D *D_projData = new CData2D(projDims[0], projDims[1], s);

	ok = astraCUDA::copyToGPUMemory(m_pSinogram, D_projData);

	if (ok)
		ok &= callBP(D_volData, D_projData, 1.0f);

	if (ok)
		ok &= astraCUDA::copyFromGPUMemory(m_pReconstruction, D_volData);

	astraCUDA::freeGPUMemory(D_volData);
	astraCUDA::freeGPUMemory(D_projData);
	delete D_volData;
	delete D_projData;

	return ok;
}


} // namespace astra

#endif // ASTRA_CUDA
