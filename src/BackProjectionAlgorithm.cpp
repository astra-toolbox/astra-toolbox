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

#include "astra/BackProjectionAlgorithm.h"

#include "astra/AstraObjectManager.h"
#include "astra/DataProjectorPolicies.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CBackProjectionAlgorithm::type = "BP";

//----------------------------------------------------------------------------------------
// Constructor
CBackProjectionAlgorithm::CBackProjectionAlgorithm() 
{
	_clear();
}

//---------------------------------------------------------------------------------------
// Initialize - C++
CBackProjectionAlgorithm::CBackProjectionAlgorithm(CProjector2D* _pProjector, 
							   CFloat32ProjectionData2D* _pSinogram, 
							   CFloat32VolumeData2D* _pReconstruction)
{
	_clear();
	initialize(_pProjector, _pSinogram, _pReconstruction);
}

//----------------------------------------------------------------------------------------
// Destructor
CBackProjectionAlgorithm::~CBackProjectionAlgorithm() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CBackProjectionAlgorithm::_clear()
{
	CReconstructionAlgorithm2D::_clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CBackProjectionAlgorithm::clear()
{
	CReconstructionAlgorithm2D::_clear();
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check
bool CBackProjectionAlgorithm::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm2D::_check(), "BP", "Error in ReconstructionAlgorithm2D initialization");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CBackProjectionAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("BackProjectionAlgorithm", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CReconstructionAlgorithm2D::initialize(_cfg)) {
		return false;
	}

	// init data objects and data projectors
	_init();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CBackProjectionAlgorithm::initialize(CProjector2D* _pProjector, 
								CFloat32ProjectionData2D* _pSinogram, 
								CFloat32VolumeData2D* _pReconstruction)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// required classes
	m_pProjector = _pProjector;
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	// init data objects and data projectors
	_init();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize Data Projectors - private
void CBackProjectionAlgorithm::_init()
{

}

//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CBackProjectionAlgorithm::getInformation() 
{
	map<string, boost::any> res;
	return mergeMap<string,boost::any>(CReconstructionAlgorithm2D::getInformation(), res);
};

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CBackProjectionAlgorithm::getInformation(std::string _sIdentifier) 
{
	return CAlgorithm::getInformation(_sIdentifier);
};

//----------------------------------------------------------------------------------------
// Iterate
void CBackProjectionAlgorithm::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	m_bShouldAbort = false;

	CDataProjectorInterface* pBackProjector;

	pBackProjector = dispatchDataProjector(
			m_pProjector, 
			SinogramMaskPolicy(m_pSinogramMask),														// sinogram mask
			ReconstructionMaskPolicy(m_pReconstructionMask),											// reconstruction mask
			DefaultBPPolicy(m_pReconstruction, m_pSinogram), // backprojection
			m_bUseSinogramMask, m_bUseReconstructionMask, true // options on/off
		); 

	m_pReconstruction->setData(0.0f);
	pBackProjector->project();

	ASTRA_DELETE(pBackProjector);
}
//----------------------------------------------------------------------------------------

} // namespace astra
