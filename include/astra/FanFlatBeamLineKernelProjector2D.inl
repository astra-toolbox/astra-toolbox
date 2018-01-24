/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#define policy_weight(p,rayindex,volindex,weight) do { if (p.pixelPrior(volindex)) { p.addWeight(rayindex, volindex, weight); p.pixelPosterior(volindex); } } while (false)

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
	// get vector geometry
	const CFanFlatVecProjectionGeometry2D* pVecProjectionGeometry;
	if (dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry)) {
		pVecProjectionGeometry = dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry)->toVectorGeometry();
	} else {
		pVecProjectionGeometry = dynamic_cast<CFanFlatVecProjectionGeometry2D*>(m_pProjectionGeometry);
	}

	// precomputations
	const float32 pixelLengthX = m_pVolumeGeometry->getPixelLengthX();
	const float32 pixelLengthY = m_pVolumeGeometry->getPixelLengthY();
	const float32 inv_pixelLengthX = 1.0f / pixelLengthX;
	const float32 inv_pixelLengthY = 1.0f / pixelLengthY;
	const int colCount = m_pVolumeGeometry->getGridColCount();
	const int rowCount = m_pVolumeGeometry->getGridRowCount();
	const int detCount = pVecProjectionGeometry->getDetectorCount();
	const float32 Ex = m_pVolumeGeometry->getWindowMinX() + pixelLengthX*0.5f;
	const float32 Ey = m_pVolumeGeometry->getWindowMaxY() - pixelLengthY*0.5f;

	// loop angles
	for (int iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		// variables
		float32 Dx, Dy, Rx, Ry, S, T, weight, c, r, deltac, deltar, offset, RxOverRy, RyOverRx;
		float32 lengthPerRow, lengthPerCol, invTminSTimesLengthPerRow, invTminSTimesLengthPerCol;
		int iVolumeIndex, iRayIndex, row, col, iDetector;

		const SFanProjection * proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		float32 detSize = sqrt(proj->fDetUX * proj->fDetUX + proj->fDetUY * proj->fDetUY);

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * detCount + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			Dx = proj->fDetSX + (iDetector+0.5f) * proj->fDetUX;
			Dy = proj->fDetSY + (iDetector+0.5f) * proj->fDetUY;

			Rx = proj->fSrcX - Dx;
			Ry = proj->fSrcY - Dy;

			bool vertical = fabs(Rx) < fabs(Ry);
			bool isin = false;

			// vertically
			if (vertical) {
				RxOverRy = Rx/Ry;
				lengthPerRow = detSize * pixelLengthX * sqrt(Rx*Rx + Ry*Ry) / abs(Ry);
				deltac = -pixelLengthY * RxOverRy * inv_pixelLengthX;
				S = 0.5f - 0.5f*fabs(RxOverRy);
				T = 0.5f + 0.5f*fabs(RxOverRy);
				invTminSTimesLengthPerRow = lengthPerRow / (T - S);

				// calculate c for row 0
				c = (Dx + (Ey - Dy)*RxOverRy - Ex) * inv_pixelLengthX;

				// for each row
				for (row = 0; row < rowCount; ++row, c += deltac) {

					col = int(floor(c+0.5f));
					if (col < -1 || col > colCount) { if (!isin) continue; else break; }
					offset = c - float32(col);

					// left
					if (offset < -S) {
						weight = (offset + T) * invTminSTimesLengthPerRow;

						iVolumeIndex = row * colCount + col - 1;
						if (col > 0) { policy_weight(p, iRayIndex, iVolumeIndex, lengthPerRow-weight); }

						iVolumeIndex++;
						if (col >= 0 && col < colCount) { policy_weight(p, iRayIndex, iVolumeIndex, weight); }
					}

					// right
					else if (S < offset) {
						weight = (offset - S) * invTminSTimesLengthPerRow;

						iVolumeIndex = row * colCount + col;
						if (col >= 0 && col < colCount) { policy_weight(p, iRayIndex, iVolumeIndex, lengthPerRow-weight); }

						iVolumeIndex++;
						if (col + 1 < colCount) { policy_weight(p, iRayIndex, iVolumeIndex, weight); } 
					}

					// centre
					else if (col >= 0 && col < colCount) {
						iVolumeIndex = row * colCount + col;
						policy_weight(p, iRayIndex, iVolumeIndex, lengthPerRow);
					}
					isin = true;
				}
			}

			// horizontally
			else {
				RyOverRx = Ry/Rx;
				lengthPerCol = detSize * pixelLengthY * sqrt(Rx*Rx + Ry*Ry) / abs(Rx);
				deltar = -pixelLengthX * RyOverRx * inv_pixelLengthY;
				S = 0.5f - 0.5f*fabs(RyOverRx);
				T = 0.5f + 0.5f*fabs(RyOverRx);
				invTminSTimesLengthPerCol = lengthPerCol / (T - S);

				// calculate r for col 0
				r = -(Dy + (Ex - Dx)*RyOverRx - Ey) * inv_pixelLengthY;

				// for each col
				for (col = 0; col < colCount; ++col, r += deltar) {

					row = int(floor(r+0.5f));
					if (row < -1 || row > rowCount) { if (!isin) continue; else break; }
					offset = r - float32(row);

					// up
					if (offset < -S) {
						weight = (offset + T) * invTminSTimesLengthPerCol;

						iVolumeIndex = (row-1) * colCount + col;
						if (row > 0) { policy_weight(p, iRayIndex, iVolumeIndex, lengthPerCol-weight); }

						iVolumeIndex += colCount;
						if (row >= 0 && row < rowCount) { policy_weight(p, iRayIndex, iVolumeIndex, weight); }
					}

					// down
					else if (S < offset) {
						weight = (offset - S) * invTminSTimesLengthPerCol;

						iVolumeIndex = row * colCount + col;
						if (row >= 0 && row < rowCount) { policy_weight(p, iRayIndex, iVolumeIndex, lengthPerCol-weight); }

						iVolumeIndex += colCount;
						if (row + 1 < rowCount) { policy_weight(p, iRayIndex, iVolumeIndex, weight); }
					}

					// centre
					else if (row >= 0 && row < rowCount) {
						iVolumeIndex = row * colCount + col;
						policy_weight(p, iRayIndex, iVolumeIndex, lengthPerCol);
					}
					isin = true;
				}
			}
	
			// POLICY: RAY POSTERIOR
			p.rayPosterior(iRayIndex);
	
		} // end loop detector
	} // end loop angles

	// Delete created vec geometry if required
	if (dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry))
		delete pVecProjectionGeometry;

}
