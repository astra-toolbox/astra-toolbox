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

#include "astra/FilteredBackProjectionAlgorithm.h"

#include <iostream>
#include <iomanip>
#include <math.h>

#include "astra/AstraObjectManager.h"
#include "astra/ParallelBeamLineKernelProjector2D.h"
#include "astra/Fourier.h"
#include "astra/DataProjector.h"

#include "astra/Logging.h"

using namespace std;

namespace astra {

#include "astra/Projector2DImpl.inl"

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CFilteredBackProjectionAlgorithm::type = "FBP";
const int FFT = 1;
const int IFFT = -1;

//----------------------------------------------------------------------------------------
// Constructor
CFilteredBackProjectionAlgorithm::CFilteredBackProjectionAlgorithm() 
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CFilteredBackProjectionAlgorithm::~CFilteredBackProjectionAlgorithm() 
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CFilteredBackProjectionAlgorithm::_clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CFilteredBackProjectionAlgorithm::clear()
{
	m_pProjector = NULL;
	m_pSinogram = NULL;
	m_pReconstruction = NULL;
	m_bIsInitialized = false;
}


//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CFilteredBackProjectionAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	
	// projector
	XMLNode node = _cfg.self.getSingleNode("ProjectorId");
	ASTRA_CONFIG_CHECK(node, "FilteredBackProjection", "No ProjectorId tag specified.");
	int id = node.getContentInt();
	m_pProjector = CProjector2DManager::getSingleton().get(id);

	// sinogram data
	node = _cfg.self.getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "FilteredBackProjection", "No ProjectionDataId tag specified.");
	id = node.getContentInt();
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));

	// volume data
	node = _cfg.self.getSingleNode("ReconstructionDataId");
	ASTRA_CONFIG_CHECK(node, "FilteredBackProjection", "No ReconstructionDataId tag specified.");
	id = node.getContentInt();
	m_pReconstruction = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));

	node = _cfg.self.getSingleNode("ProjectionIndex");
	if (node) 
	{
		vector<float32> projectionIndex = node.getContentNumericalArray();

		int angleCount = projectionIndex.size();
		int detectorCount = m_pProjector->getProjectionGeometry()->getDetectorCount();

		// TODO: There is no need to allocate this. Better just
		// create the CFloat32ProjectionData2D object directly, and use its
		// memory.
		float32 * sinogramData2D = new float32[angleCount* detectorCount];

		float32 * projectionAngles = new float32[angleCount];
		float32 detectorWidth = m_pProjector->getProjectionGeometry()->getDetectorWidth();

		for (int i = 0; i < angleCount; i ++) {
			if (projectionIndex[i] > m_pProjector->getProjectionGeometry()->getProjectionAngleCount() -1 )
			{
				delete[] sinogramData2D;
				delete[] projectionAngles;
				ASTRA_ERROR("Invalid Projection Index");
				return false;
			} else {
				int orgIndex = (int)projectionIndex[i];

				for (int iDetector=0; iDetector < detectorCount; iDetector++) 
				{
					sinogramData2D[i*detectorCount+ iDetector] = m_pSinogram->getData2D()[orgIndex][iDetector];
				}
				projectionAngles[i] = m_pProjector->getProjectionGeometry()->getProjectionAngle((int)projectionIndex[i] );

			}
		}

		CParallelProjectionGeometry2D * pg = new CParallelProjectionGeometry2D(angleCount, detectorCount,detectorWidth,projectionAngles);
		m_pProjector = new CParallelBeamLineKernelProjector2D(pg,m_pReconstruction->getGeometry());
		m_pSinogram = new CFloat32ProjectionData2D(pg, sinogramData2D);

		delete[] sinogramData2D;
		delete[] projectionAngles;
	}

	// TODO: check that the angles are linearly spaced between 0 and pi

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Get Information - all
map<string,boost::any> CFilteredBackProjectionAlgorithm::getInformation() 
{
	map<string, boost::any> result;
	result["ProjectorId"] = getInformation("ProjectorId");
	result["ProjectionDataId"] = getInformation("ProjectionDataId");
	result["VolumeDataId"] = getInformation("VolumeDataId");
	return mergeMap<string,boost::any>(CAlgorithm::getInformation(), result);
};

//---------------------------------------------------------------------------------------
// Get Information - specific
boost::any CFilteredBackProjectionAlgorithm::getInformation(std::string _sIdentifier) 
{
	if (_sIdentifier == "ProjectorId") {
		int iIndex = CProjector2DManager::getSingleton().getIndex(m_pProjector);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	} else if (_sIdentifier == "ProjectionDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pSinogram);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	} else if (_sIdentifier == "VolumeDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pReconstruction);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	return CAlgorithm::getInformation(_sIdentifier);
};

//----------------------------------------------------------------------------------------
// Initialize
bool CFilteredBackProjectionAlgorithm::initialize(CProjector2D* _pProjector, 
												  CFloat32VolumeData2D* _pVolume,
												  CFloat32ProjectionData2D* _pSinogram)
{
	// store classes
	m_pProjector = _pProjector;
	m_pReconstruction = _pVolume;
	m_pSinogram = _pSinogram;


	// TODO: check that the angles are linearly spaced between 0 and pi

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Check
bool CFilteredBackProjectionAlgorithm::_check() 
{
	ASTRA_CONFIG_CHECK(CReconstructionAlgorithm2D::_check(), "FBP", "Error in ReconstructionAlgorithm2D initialization");

	// success
	return true;
}


//----------------------------------------------------------------------------------------
// Iterate
void CFilteredBackProjectionAlgorithm::run(int _iNrIterations)
{
	ASTRA_ASSERT(m_bIsInitialized);

	// Filter sinogram
	CFloat32ProjectionData2D filteredSinogram(m_pSinogram->getGeometry(), m_pSinogram->getData());
	performFiltering(&filteredSinogram);

	// Back project
	m_pReconstruction->setData(0.0f);
	projectData(m_pProjector,
	            DefaultBPPolicy(m_pReconstruction, &filteredSinogram));

	// Scale data
	int iAngleCount = m_pProjector->getProjectionGeometry()->getProjectionAngleCount();
	(*m_pReconstruction) *= (PI/2)/iAngleCount;

	m_pReconstruction->updateStatistics();
}


//----------------------------------------------------------------------------------------
void CFilteredBackProjectionAlgorithm::performFiltering(CFloat32ProjectionData2D * _pFilteredSinogram)
{
	ASTRA_ASSERT(_pFilteredSinogram != NULL);
	ASTRA_ASSERT(_pFilteredSinogram->getAngleCount() == m_pSinogram->getAngleCount());
	ASTRA_ASSERT(_pFilteredSinogram->getDetectorCount() == m_pSinogram->getDetectorCount());


	int iAngleCount = m_pProjector->getProjectionGeometry()->getProjectionAngleCount();
	int iDetectorCount = m_pProjector->getProjectionGeometry()->getDetectorCount();


	// We'll zero-pad to the smallest power of two at least 64 and
	// at least 2*iDetectorCount
	int zpDetector = 64;
	int nextPow2 = 5;
	while (zpDetector < iDetectorCount*2) {
		zpDetector *= 2;
		nextPow2++;
	}

	// Create filter
	float32* filter = new float32[zpDetector];

	for (int iDetector = 0; iDetector <= zpDetector/2; iDetector++)
		filter[iDetector] = (2.0f * iDetector)/zpDetector;

	for (int iDetector = zpDetector/2+1; iDetector < zpDetector; iDetector++)
		filter[iDetector] = (2.0f * (zpDetector - iDetector)) / zpDetector;


	float32* pf = new float32[2 * iAngleCount * zpDetector];
	int *ip = new int[int(2+sqrt((float)zpDetector)+1)];
	ip[0]=0;
	float32 *w = new float32[zpDetector/2];

	// Copy and zero-pad data
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		float32* pfDataRow = _pFilteredSinogram->getData() + iAngle * iDetectorCount;
		for (int iDetector = 0; iDetector < iDetectorCount; ++iDetector) {
			pfRow[2*iDetector] = pfDataRow[iDetector];
			pfRow[2*iDetector+1] = 0.0f;
		}
		for (int iDetector = iDetectorCount; iDetector < zpDetector; ++iDetector) {
			pfRow[2*iDetector] = 0.0f;
			pfRow[2*iDetector+1] = 0.0f;
		}
	}

	// in-place FFT
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		cdft(2*zpDetector, -1, pfRow, ip, w);
	}

	// Filter
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		for (int iDetector = 0; iDetector < zpDetector; ++iDetector) {
			pfRow[2*iDetector] *= filter[iDetector];
			pfRow[2*iDetector+1] *= filter[iDetector];
		}
	}

	// in-place inverse FFT
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		cdft(2*zpDetector, 1, pfRow, ip, w);
	}

	// Copy data back
	for (int iAngle = 0; iAngle < iAngleCount; ++iAngle) {
		float32* pfRow = pf + iAngle * 2 * zpDetector;
		float32* pfDataRow = _pFilteredSinogram->getData() + iAngle * iDetectorCount;
		for (int iDetector = 0; iDetector < iDetectorCount; ++iDetector)
			pfDataRow[iDetector] = pfRow[2*iDetector] / zpDetector;
	}

	delete[] pf;
	delete[] w;
	delete[] ip;
	delete[] filter;
}

}
