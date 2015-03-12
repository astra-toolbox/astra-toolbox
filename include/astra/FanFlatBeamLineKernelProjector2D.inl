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


using namespace astra;

template <typename Policy>
void CFanFlatBeamLineKernelProjector2D::project(Policy& p)
{
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CFanFlatBeamLineKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CFanFlatBeamLineKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK - vector projection geometry
template <typename Policy>
void CFanFlatBeamLineKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 detX, detY, S, T, I, x, y, c, r, update_c, update_r, offset;
	float32 lengthPerRow, lengthPerCol, inv_pixelLengthX, inv_pixelLengthY, invTminSTimesLengthPerRow, invTminSTimesLengthPerCol;
	int iVolumeIndex, iRayIndex, row, col, iAngle, iDetector, colCount, rowCount, detCount;
	const SFanProjection * proj = 0;

	// get vector geometry
	const CFanFlatVecProjectionGeometry2D* pVecProjectionGeometry;
	if (dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry)) {
		pVecProjectionGeometry = dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry)->toVectorGeometry();
	} else {
		pVecProjectionGeometry = dynamic_cast<CFanFlatVecProjectionGeometry2D*>(m_pProjectionGeometry);
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

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * detCount + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			detX = proj->fDetSX + (iDetector+0.5f) * proj->fDetUX;
			detY = proj->fDetSY + (iDetector+0.5f) * proj->fDetUY;

			float32 fRayX = proj->fSrcX - detX;
			float32 fRayY = proj->fSrcY - detY;

			bool vertical = fabs(fRayX) < fabs(fRayY);
			if (vertical) {
				lengthPerRow = m_pVolumeGeometry->getPixelLengthX() * sqrt(fRayY*fRayY + fRayX*fRayX) / abs(fRayY);
				update_c = -m_pVolumeGeometry->getPixelLengthY() * (fRayX/fRayY) * inv_pixelLengthX;
				S = 0.5f - 0.5f*fabs(fRayX/fRayY);
				T = 0.5f + 0.5f*fabs(fRayX/fRayY);
				invTminSTimesLengthPerRow = lengthPerRow / (T - S);
			} else {
				lengthPerCol = m_pVolumeGeometry->getPixelLengthY() * sqrt(fRayY*fRayY + fRayX*fRayX) / abs(fRayX);
				update_r = -m_pVolumeGeometry->getPixelLengthX() * (fRayY/fRayX) * inv_pixelLengthY;
				S = 0.5f - 0.5f*fabs(fRayY/fRayX);
				T = 0.5f + 0.5f*fabs(fRayY/fRayX);
				invTminSTimesLengthPerCol = lengthPerCol / (T - S);
			}

			// vertically
			if (vertical) {

				// calculate x for row 0
				x = detX + (fRayX/fRayY)*(m_pVolumeGeometry->pixelRowToCenterY(0)-detY);
				c = (x - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f;

				// for each row
				for (row = 0; row < rowCount; ++row, c += update_c) {

					col = int(c+0.5f);
					offset = c - float32(col);

					if (col <= 0 || col >= colCount-1) continue;

					// left
					if (offset < -S) {
						I = (offset + T) * invTminSTimesLengthPerRow;

						iVolumeIndex = row * colCount + col - 1;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, lengthPerRow-I);
							p.pixelPosterior(iVolumeIndex);
						}

						iVolumeIndex++;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, I);
							p.pixelPosterior(iVolumeIndex);
						}
					}

					// right
					else if (S < offset) {
						I = (offset - S) * invTminSTimesLengthPerRow;

						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, lengthPerRow-I);
							p.pixelPosterior(iVolumeIndex);
						}

						iVolumeIndex++;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, I);
							p.pixelPosterior(iVolumeIndex);
						}
					}

					// centre
					else {
						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, lengthPerRow);
							p.pixelPosterior(iVolumeIndex);
						}
					}
		
				}
			}

			// horizontally
			else {

				// calculate y for col 0
				y = detY + (fRayY/fRayX)*(m_pVolumeGeometry->pixelColToCenterX(0)-detX);
				r = (m_pVolumeGeometry->getWindowMaxY() - y) * inv_pixelLengthY - 0.5f;

				// for each col
				for (col = 0; col < colCount; ++col, r += update_r) {

					int row = int(r+0.5f);
					offset = r - float32(row);

					if (row <= 0 || row >= rowCount-1) continue;

					// up
					if (offset < -S) {
						I = (offset + T) * invTminSTimesLengthPerCol;

						iVolumeIndex = (row-1) * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, lengthPerCol-I);
							p.pixelPosterior(iVolumeIndex);
						}

						iVolumeIndex += colCount;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, I);
							p.pixelPosterior(iVolumeIndex);
						}
					}

					// down
					else if (S < offset) {
						I = (offset - S) * invTminSTimesLengthPerCol;

						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, lengthPerCol-I);
							p.pixelPosterior(iVolumeIndex);
						}

						iVolumeIndex += colCount;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, I);
							p.pixelPosterior(iVolumeIndex);
						}
					}

					// centre
					else {
						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, lengthPerCol);
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
