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


//----------------------------------------------------------------------------------------
// PROJECT ALL 
template <typename Policy>
void CSparseMatrixProjector2D::project(Policy& p)
{
	ASTRA_ASSERT(m_bIsInitialized);

	for (int i = 0; i < m_pProjectionGeometry->getProjectionAngleCount(); ++i)
		for (int j = 0; j < m_pProjectionGeometry->getDetectorCount(); ++j)
			projectSingleRay(i, j, p);
}


//----------------------------------------------------------------------------------------
// PROJECT SINGLE PROJECTION
template <typename Policy>
void CSparseMatrixProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	ASTRA_ASSERT(m_bIsInitialized);

	for (int j = 0; j < m_pProjectionGeometry->getDetectorCount(); ++j)
		projectSingleRay(_iProjection, j, p);
}


//----------------------------------------------------------------------------------------
// PROJECT SINGLE RAY
template <typename Policy>
void CSparseMatrixProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	ASTRA_ASSERT(m_bIsInitialized);

	int iRayIndex = _iProjection * m_pProjectionGeometry->getDetectorCount() + _iDetector;
	const CSparseMatrix* pMatrix = dynamic_cast<CSparseMatrixProjectionGeometry2D*>(m_pProjectionGeometry)->getMatrix();

	// POLICY: RAY PRIOR
	if (!p.rayPrior(iRayIndex)) return;

	const unsigned int* piColIndices;
	const float32* pfValues;
	unsigned int iSize;

	pMatrix->getRowData(iRayIndex, iSize, pfValues, piColIndices);

	for (unsigned int i = 0; i < iSize; ++i) {
		unsigned int iVolumeIndex = piColIndices[i];

		// POLICY: PIXEL PRIOR
		if (p.pixelPrior(iVolumeIndex)) {
				
			// POLICY: ADD
			p.addWeight(iRayIndex, iVolumeIndex, pfValues[i]);

			// POLICY: PIXEL POSTERIOR
			p.pixelPosterior(iVolumeIndex);
		}
	}

	// POLICY: RAY POSTERIOR
	p.rayPosterior(iRayIndex);
}
