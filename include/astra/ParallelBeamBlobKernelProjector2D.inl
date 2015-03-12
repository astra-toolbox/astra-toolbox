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
void CParallelBeamBlobKernelProjector2D::project(Policy& p)
{
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
		                  0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK - vector projection geometry
template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 detX, detY, x, y, c, r, update_c, update_r, offset, invBlobExtent, inv_pixelLengthX, inv_pixelLengthY, weight, d;
	int iVolumeIndex, iRayIndex, row, col, iAngle, iDetector, colCount, rowCount, detCount;
	int col_left, col_right, row_top, row_bottom, index;
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
			update_c = -m_pVolumeGeometry->getPixelLengthY() * (proj->fRayX/proj->fRayY) * inv_pixelLengthX;
			float32 normR = sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX);
			invBlobExtent = m_pVolumeGeometry->getPixelLengthY() / abs(m_fBlobSize * normR / proj->fRayY);
		} else {
			update_r = -m_pVolumeGeometry->getPixelLengthX() * (proj->fRayY/proj->fRayX) * inv_pixelLengthY;
			float32 normR = sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX);
			invBlobExtent = m_pVolumeGeometry->getPixelLengthX() / abs(m_fBlobSize * normR / proj->fRayX);
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

					col_left = int(c - 0.5f - m_fBlobSize);
					col_right = int(c + 0.5f + m_fBlobSize);

					if (col_left < 0) col_left = 0; 
					if (col_right > colCount-1) col_right = colCount-1; 

					// for each column
					for (col = col_left; col <= col_right; ++col) {

						offset = abs(c - float32(col)) * invBlobExtent;
						index = (int)(offset*m_iBlobSampleCount+0.5f);
						weight = m_pfBlobValues[min(index,m_iBlobSampleCount-1)];
				
						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, weight);
							p.pixelPosterior(iVolumeIndex);
						}
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

					row_top = int(r - 0.5f - m_fBlobSize);
					row_bottom = int(r + 0.5f + m_fBlobSize);

					if (row_top < 0) row_top = 0; 
					if (row_bottom > rowCount-1) row_bottom = rowCount-1; 

					// for each column
					for (row = row_top; row <= row_bottom; ++row) {

						offset = abs(r - float32(row)) * invBlobExtent;
						index = (int)(offset*m_iBlobSampleCount+0.5f);
						weight = m_pfBlobValues[min(index,m_iBlobSampleCount-1)];
				
						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, weight);
							p.pixelPosterior(iVolumeIndex);
						}
					}

				}
			}
	
			// POLICY: RAY POSTERIOR
			p.rayPosterior(iRayIndex);
	
		} // end loop detector
	} // end loop angles

}
