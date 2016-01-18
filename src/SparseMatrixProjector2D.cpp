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

#include "astra/SparseMatrixProjector2D.h"

#include <cmath>

#include "astra/DataProjectorPolicies.h"

using namespace std;
using namespace astra;

#include "astra/SparseMatrixProjector2D.inl"

// type of the projector, needed to register with CProjectorFactory
std::string CSparseMatrixProjector2D::type = "sparse_matrix";

//----------------------------------------------------------------------------------------
// default constructor
CSparseMatrixProjector2D::CSparseMatrixProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// constructor
CSparseMatrixProjector2D::CSparseMatrixProjector2D(CSparseMatrixProjectionGeometry2D* _pProjectionGeometry,
                                                   CVolumeGeometry2D* _pReconstructionGeometry)

{
	_clear();
	initialize(_pProjectionGeometry, _pReconstructionGeometry);
}

//----------------------------------------------------------------------------------------
// destructor
CSparseMatrixProjector2D::~CSparseMatrixProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CSparseMatrixProjector2D::_clear()
{
	CProjector2D::_clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CSparseMatrixProjector2D::clear()
{
	CProjector2D::clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CSparseMatrixProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "SparseMatrixProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "SparseMatrixProjector2D", "Volume geometry not initialized");
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "SparseMatrixProjector2D", "Projection geometry not initialized");
	

	ASTRA_CONFIG_CHECK(dynamic_cast<CSparseMatrixProjectionGeometry2D*>(m_pProjectionGeometry), "SparseMatrixProjector2D", "Unsupported projection geometry");

	const CSparseMatrix* pMatrix = dynamic_cast<CSparseMatrixProjectionGeometry2D*>(m_pProjectionGeometry)->getMatrix();
	ASTRA_CONFIG_CHECK(pMatrix, "SparseMatrixProjector2D", "No matrix specified in projection geometry");

	ASTRA_CONFIG_CHECK(pMatrix->m_iWidth == (unsigned int)m_pVolumeGeometry->getGridTotCount(), "SparseMatrixProjector2D", "Matrix width doesn't match volume geometry");

	return true;
}


//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CSparseMatrixProjector2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector2D::initialize(_cfg)) {
		return false;
	}

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize
bool CSparseMatrixProjector2D::initialize(CSparseMatrixProjectionGeometry2D* _pProjectionGeometry, 
													 CVolumeGeometry2D* _pVolumeGeometry)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// hardcopy geometries
	m_pProjectionGeometry = _pProjectionGeometry->clone();
	m_pVolumeGeometry = _pVolumeGeometry->clone();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}


//----------------------------------------------------------------------------------------
// Get maximum amount of weights on a single ray
int CSparseMatrixProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	const CSparseMatrix* pMatrix = dynamic_cast<CSparseMatrixProjectionGeometry2D*>(m_pProjectionGeometry)->getMatrix();

	unsigned int iMax = 0;
	unsigned long lSize = m_pProjectionGeometry->getDetectorCount();
	lSize *= m_pProjectionGeometry->getProjectionAngleCount();
	for (unsigned long i = 0; i < lSize; ++i) {
		unsigned int iRowSize = pMatrix->getRowSize(i);
		if (iRowSize > iMax)
			iMax = pMatrix->getRowSize(i);
	}
	return iMax;
}

//----------------------------------------------------------------------------------------
// Single Ray Weights
void CSparseMatrixProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
																  int _iDetectorIndex, 
																  SPixelWeight* _pWeightedPixels,
																  int _iMaxPixelCount, 
																  int& _iStoredPixelCount)
{
	// TODO: Move this default implementation to Projector2D?
	ASTRA_ASSERT(m_bIsInitialized);
	StorePixelWeightsPolicy p(_pWeightedPixels, _iMaxPixelCount);
	projectSingleRay(_iProjectionIndex, _iDetectorIndex, p);
	_iStoredPixelCount = p.getStoredPixelCount();
}

//----------------------------------------------------------------------------------------
// Splat a single point
std::vector<SDetector2D> CSparseMatrixProjector2D::projectPoint(int _iRow, int _iCol)
{
	unsigned int iVolumeIndex = _iCol * m_pVolumeGeometry->getGridRowCount() + _iRow;

	// NOTE: This is very slow currently because we don't have the
	// sparse matrix stored in an appropriate form for this function.
	std::vector<SDetector2D> ret;

	const CSparseMatrix* pMatrix = dynamic_cast<CSparseMatrixProjectionGeometry2D*>(m_pProjectionGeometry)->getMatrix();

	for (int iAngle = 0; iAngle < m_pProjectionGeometry->getProjectionAngleCount(); ++iAngle)
	{
		for (int iDetector = 0; iDetector < m_pProjectionGeometry->getDetectorCount(); ++iDetector)
		{
			int iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;
			const unsigned int* piColIndices;
			const float32* pfValues;
			unsigned int iSize;
    
			pMatrix->getRowData(iRayIndex, iSize, pfValues, piColIndices);

			for (unsigned int i = 0; i < iSize; ++i) {
				if (piColIndices[i] == iVolumeIndex) {
					SDetector2D s;
					s.m_iIndex = iRayIndex;
					s.m_iAngleIndex = iAngle;
					s.m_iDetectorIndex = iDetector;
					ret.push_back(s);
					break;
				} else if (piColIndices[i] > iVolumeIndex) {
					break;
				}
			}
		}
	}
	return ret;
}

//----------------------------------------------------------------------------------------
