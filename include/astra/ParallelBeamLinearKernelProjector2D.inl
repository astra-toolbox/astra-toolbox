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

template <typename Policy>
void CParallelBeamLinearKernelProjector2D::project(Policy& p)
{
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
		                  0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamLinearKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamLinearKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
}



//----------------------------------------------------------------------------------------
// PROJECT BLOCK - vector projection geometry
template <typename Policy>
void CParallelBeamLinearKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 detX, detY, x, y, c, r, update_c, update_r, offset;
	float32 lengthPerRow, lengthPerCol, inv_pixelLengthX, inv_pixelLengthY;
	int iVolumeIndex, iRayIndex, row, col, iAngle, iDetector, colCount, rowCount, detCount;
	const SParProjection * proj = 0;

	// get vector geometry
	const CParallelVecProjectionGeometry2D* pVecProjectionGeometry;
	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)) {
		pVecProjectionGeometry = dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)->toVectorGeometry();
	} else {
		pVecProjectionGeometry = dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry);
	}

	// precomputations
	inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();
	inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();
	colCount = m_pVolumeGeometry->getGridColCount();
	rowCount = m_pVolumeGeometry->getGridRowCount();
	detCount = pVecProjectionGeometry->getDetectorCount();

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			lengthPerRow = m_pVolumeGeometry->getPixelLengthX() * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / abs(proj->fRayY);
			update_c = -m_pVolumeGeometry->getPixelLengthY() * (proj->fRayX/proj->fRayY) * inv_pixelLengthX;
		} else {
			lengthPerCol = m_pVolumeGeometry->getPixelLengthY() * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / abs(proj->fRayX);
			update_r = -m_pVolumeGeometry->getPixelLengthX() * (proj->fRayY/proj->fRayX) * inv_pixelLengthY;
		}

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			detX = proj->fDetSX + (iDetector+0.5f) * proj->fDetUX;
			detY = proj->fDetSY + (iDetector+0.5f) * proj->fDetUY;

			// vertically
			if (vertical) {

				// calculate x for row 0
				x = detX + (proj->fRayX/proj->fRayY)*(m_pVolumeGeometry->pixelRowToCenterY(0)-detY);
				c = (x - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f;

				// for each row
				for (row = 0; row < rowCount; ++row, c += update_c) {

					col = int(c);
					offset = c - float32(col);

					if (col <= 0 || col >= colCount-1) continue;

					iVolumeIndex = row * colCount + col;
					// POLICY: PIXEL PRIOR + ADD + POSTERIOR
					if (p.pixelPrior(iVolumeIndex)) {
						p.addWeight(iRayIndex, iVolumeIndex, (1.0f - offset) * lengthPerRow);
						p.pixelPosterior(iVolumeIndex);
					}
					
					iVolumeIndex++;
					// POLICY: PIXEL PRIOR + ADD + POSTERIOR
					if (p.pixelPrior(iVolumeIndex)) {
						p.addWeight(iRayIndex, iVolumeIndex, offset * lengthPerRow);
						p.pixelPosterior(iVolumeIndex);
					}

				}
			}

			// horizontally
			else {

				// calculate y for col 0
				y = detY + (proj->fRayY/proj->fRayX)*(m_pVolumeGeometry->pixelColToCenterX(0)-detX);
				r = (m_pVolumeGeometry->getWindowMaxY() - y) * inv_pixelLengthY - 0.5f;

				// for each col
				for (col = 0; col < colCount; ++col, r += update_r) {

					int row = int(r);
					offset = r - float32(row);

					if (row <= 0 || row >= rowCount-1) continue;

					iVolumeIndex = row * colCount + col;
					// POLICY: PIXEL PRIOR + ADD + POSTERIOR
					if (p.pixelPrior(iVolumeIndex)) {
						p.addWeight(iRayIndex, iVolumeIndex, (1.0f - offset) * lengthPerCol);
						p.pixelPosterior(iVolumeIndex);
					}

					iVolumeIndex += colCount;
					// POLICY: PIXEL PRIOR + ADD + POSTERIOR
					if (p.pixelPrior(iVolumeIndex)) {
						p.addWeight(iRayIndex, iVolumeIndex, offset * lengthPerCol);
						p.pixelPosterior(iVolumeIndex);
					}

				}
			}
	
			// POLICY: RAY POSTERIOR
			p.rayPosterior(iRayIndex);
	
		} // end loop detector
	} // end loop angles
}
