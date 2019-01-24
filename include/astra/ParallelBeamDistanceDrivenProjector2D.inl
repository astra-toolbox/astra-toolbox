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
void CParallelBeamDistanceDrivenProjector2D::project(Policy& p)
{
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
		                  0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamDistanceDrivenProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamDistanceDrivenProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
}





template <typename Policy>
void CParallelBeamDistanceDrivenProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// get vector geometry
	const CParallelVecProjectionGeometry2D* pVecProjectionGeometry;
	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)) {
		pVecProjectionGeometry = dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)->toVectorGeometry();
	} else {
		pVecProjectionGeometry = dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry);
	}

	// precomputations
	const float32 pixelLengthX = m_pVolumeGeometry->getPixelLengthX();
	const float32 pixelLengthY = m_pVolumeGeometry->getPixelLengthY();
	const float32 inv_pixelLengthX = 1.0f / pixelLengthX;
	const float32 inv_pixelLengthY = 1.0f / pixelLengthY;
	const int colCount = m_pVolumeGeometry->getGridColCount();
	const int rowCount = m_pVolumeGeometry->getGridRowCount();

	// Performance note:
	// This is not a very well optimizated version of the distance driven
	// projector. The CPU projector model in ASTRA requires ray-driven iteration,
	// which limits re-use of intermediate computations.

	// loop angles
	for (int iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		const SParProjection * proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		const bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);

		const float32 Ex = m_pVolumeGeometry->getWindowMinX() + pixelLengthX*0.5f;
		const float32 Ey = m_pVolumeGeometry->getWindowMaxY() - pixelLengthY*0.5f;

		// loop detectors
		for (int iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {

			const int iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;

			const float32 Dx = proj->fDetSX + (iDetector+0.5f) * proj->fDetUX;
			const float32 Dy = proj->fDetSY + (iDetector+0.5f) * proj->fDetUY;

			if (vertical) {

				const float32 RxOverRy = proj->fRayX/proj->fRayY;
				const float32 lengthPerRow = m_pVolumeGeometry->getPixelLengthX() * m_pVolumeGeometry->getPixelLengthY();
				const float32 deltac = -pixelLengthY * RxOverRy * inv_pixelLengthX;
				const float32 deltad = 0.5f * fabs((proj->fDetUX - proj->fDetUY * RxOverRy) * inv_pixelLengthX);

				// calculate c for row 0
				float32 c = (Dx + (Ey - Dy)*RxOverRy - Ex) * inv_pixelLengthX + 0.5f;

				// loop rows
				for (int row = 0; row < rowCount; ++row, c+= deltac) {

					// horizontal extent of ray in center of this row:
					// [ c - deltad , c + deltad ]

					// |-gapBegin-*---|------|----*-gapEnd-|
					// * = ray extent intercepts; c - deltad and c + deltad
					// | = pixel column edges

					const int colBegin = (int)floor(c - deltad);
					const int colEnd = (int)ceil(c + deltad);

					if (colBegin >= colCount || colEnd <= 0)
						continue;

					int iVolumeIndex = row * colCount + colBegin;
					if (colBegin + 1 == colEnd) {

						if (colBegin >= 0 && colBegin < colCount)
							policy_weight(p, iRayIndex, iVolumeIndex,
							              2.0f * deltad * lengthPerRow);
					} else {

						if (colBegin >= 0) {
							const float gapBegin = (c - deltad) - (float32)colBegin;
							policy_weight(p, iRayIndex, iVolumeIndex,
							              (1.0f - gapBegin) * lengthPerRow);
						}

						const int clippedMColBegin = std::max(colBegin + 1, 0);
						const int clippedMColEnd = std::min(colEnd - 1, colCount);
						iVolumeIndex = row * colCount + clippedMColBegin;
						for (int col = clippedMColBegin; col < clippedMColEnd; ++col, ++iVolumeIndex) {
							policy_weight(p, iRayIndex, iVolumeIndex, lengthPerRow);
						}

						iVolumeIndex = row * colCount + colEnd - 1;
						if (colEnd <= colCount) {
							const float gapEnd = (float32)colEnd - (c + deltad);
							policy_weight(p, iRayIndex, iVolumeIndex,
						 	             (1.0f - gapEnd) * lengthPerRow);
						}
					}

				}
				
			} else {

				const float32 RyOverRx = proj->fRayY/proj->fRayX;
				const float32 lengthPerCol = m_pVolumeGeometry->getPixelLengthX() * m_pVolumeGeometry->getPixelLengthY();
				const float32 deltar = -pixelLengthX * RyOverRx * inv_pixelLengthY;
				const float32 deltad = 0.5f * fabs((proj->fDetUY - proj->fDetUX * RyOverRx) * inv_pixelLengthY);

				// calculate r for col 0
				float32 r = -(Dy + (Ex - Dx)*RyOverRx - Ey) * inv_pixelLengthY + 0.5f;

				// loop columns
				for (int col = 0; col < colCount; ++col, r+= deltar) {

					// vertical extent of ray in center of this column:
					// [ r - deltad , r + deltad ]

					const int rowBegin = (int)floor(r - deltad);
					const int rowEnd = (int)ceil(r + deltad);

					if (rowBegin >= rowCount || rowEnd <= 0)
						continue;

					int iVolumeIndex = rowBegin * colCount + col;
					if (rowBegin + 1 == rowEnd) {

						if (rowBegin >= 0 && rowBegin < rowCount)
							policy_weight(p, iRayIndex, iVolumeIndex,
							              2.0f * deltad * lengthPerCol);
					} else {
						if (rowBegin >= 0) {
							const float gapBegin = (r - deltad) - (float32)rowBegin;
							policy_weight(p, iRayIndex, iVolumeIndex,
							              (1.0f - gapBegin) * lengthPerCol);
						}

						const int clippedMRowBegin = std::max(rowBegin + 1, 0);
						const int clippedMRowEnd = std::min(rowEnd - 1, rowCount);
						iVolumeIndex = clippedMRowBegin * colCount + col;

						for (int row = clippedMRowBegin; row < clippedMRowEnd; ++row, iVolumeIndex += colCount) {
							policy_weight(p, iRayIndex, iVolumeIndex, lengthPerCol);
						}

						iVolumeIndex = (rowEnd - 1) * colCount + col;
						if (rowEnd <= rowCount) {
							const float gapEnd = (float32)rowEnd - (r + deltad);
							policy_weight(p, iRayIndex, iVolumeIndex,
						 	             (1.0f - gapEnd) * lengthPerCol);
						}
					}

				}

			}

			// POLICY: RAY POSTERIOR
			p.rayPosterior(iRayIndex);
	
		}
	}

	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry))
		delete pVecProjectionGeometry;
}
