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
void CParallelBeamLineKernelProjector2D::project(Policy& p)
{
	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)) {
		projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
		                      0, m_pProjectionGeometry->getDetectorCount(), p);
	} else if (dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry)) {
		projectBlock_internal_vector(0, m_pProjectionGeometry->getProjectionAngleCount(),
		                             0, m_pProjectionGeometry->getDetectorCount(), p);
	}
}

template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)) {
		projectBlock_internal(_iProjection, _iProjection + 1,
	                          0, m_pProjectionGeometry->getDetectorCount(), p);
	} else if (dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry)) {
		projectBlock_internal_vector(_iProjection, _iProjection + 1,
	                                 0, m_pProjectionGeometry->getDetectorCount(), p);
	}
}

template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)) {
		projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
	} else if (dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry)) {
		projectBlock_internal_vector(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
	}
}


//----------------------------------------------------------------------------------------
// PROJECT BLOCK - default projection geometry
template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 theta, sin_theta, cos_theta, inv_sin_theta, inv_cos_theta, S, T, t, I, P, x, x2;
	float32 lengthPerRow, updatePerRow, inv_pixelLengthX, lengthPerCol, updatePerCol, inv_pixelLengthY;
	int iVolumeIndex, iRayIndex, row, col, iAngle, iDetector, x1;
	bool switch_t;

	float32 old_theta;
	const SParProjection * proj = 0;

	const CParallelProjectionGeometry2D* pProjectionGeometry = dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry);

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		// get theta
		theta = m_pProjectionGeometry->getProjectionAngle(iAngle);
		switch_t = false;
		if (theta >= 7*PIdiv4) theta -= 2*PI;
		if (theta >= 3*PIdiv4) {
			theta -= PI;
			switch_t = true;
		}

		// precalculate sin, cos, 1/cos
		sin_theta = sin(theta);
		cos_theta = cos(theta);
		inv_sin_theta = 1.0f / sin_theta; 
		inv_cos_theta = 1.0f / cos_theta; 

		// precalculate kernel limits
		lengthPerRow = m_pVolumeGeometry->getPixelLengthY() * inv_cos_theta;
		updatePerRow = sin_theta * inv_cos_theta;
		inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();

		// precalculate kernel limits
		lengthPerCol = m_pVolumeGeometry->getPixelLengthX() * inv_sin_theta;
		updatePerCol = cos_theta * inv_sin_theta;
		inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();

		// precalculate S and T
		S = 0.5f - 0.5f * ((updatePerRow < 0) ? -updatePerRow : updatePerRow);
		T = 0.5f - 0.5f * ((updatePerCol < 0) ? -updatePerCol : updatePerCol);

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			// get t
			t = m_pProjectionGeometry->indexToDetectorOffset(iDetector);
			if (switch_t) t = -t;

			// vertically
			if (theta <= PIdiv4) {
			
				// calculate x for row 0
				P = (t - sin_theta * m_pVolumeGeometry->pixelRowToCenterY(0)) * inv_cos_theta;
				x = (P - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX;

				// get coords
				int nextx1 = int((x > 0.0f) ? x : x-1.0f);
				float nextx2 = x - nextx1;

				// for each row
				for (row = 0; row < m_pVolumeGeometry->getGridRowCount(); ++row) {

					x1 = nextx1;
					x2 = nextx2;

					nextx2 += updatePerRow;
					while (nextx2 >= 1.0f) {
						nextx2 -= 1.0f;
						nextx1++;
					}
					while (nextx2 < 0.0f) {
						nextx2 += 1.0f;
						nextx1--;
					}

					if (x1 < -1 || x1 > m_pVolumeGeometry->getGridColCount()) continue;

					// left
					if (x2 < 0.5f-S) {
						I = (0.5f - S + x2) / (1.0f - 2.0f*S) * lengthPerRow;

						if (x1-1 >= 0 && x1-1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1-1);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, lengthPerRow-I);
								p.pixelPosterior(iVolumeIndex);
							}
						}

						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
					}

					// center
					else if (x2 <= 0.5f+S) {
						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, lengthPerRow);
								p.pixelPosterior(iVolumeIndex);
							}
						}					
					}

					// right
					else  {
						I = (1.5f - S - x2) / (1.0f - 2.0f*S) * lengthPerRow;

						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
						if (x1+1 >= 0 && x1+1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1+1);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, lengthPerRow-I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
					}
				}
			}

			// horizontally
			else if (PIdiv4 <= theta && theta <= 3*PIdiv4) {

				// calculate point P
				P = (t - cos_theta * m_pVolumeGeometry->pixelColToCenterX(0)) * inv_sin_theta;
				x = (m_pVolumeGeometry->getWindowMaxY() - P) * inv_pixelLengthY;

				// get coords
				int nextx1 = int((x > 0.0f) ? x : x-1.0f);
				float nextx2 = x - nextx1;

				// for each col
				for (col = 0; col < m_pVolumeGeometry->getGridColCount(); ++col) {

					x1 = nextx1;
					x2 = nextx2;

					nextx2 += updatePerCol;
					while (nextx2 >= 1.0f) {
						nextx2 -= 1.0f;
						nextx1++;
					}
					while (nextx2 < 0.0f) {
						nextx2 += 1.0f;
						nextx1--;
					}

					if (x1 < -1 || x1 > m_pVolumeGeometry->getGridRowCount()) continue;

					// up
					if (x2 < 0.5f-T) {
						I = (0.5f - T + x2) / (1.0f - 2.0f*T) * lengthPerCol;

						if (x1-1 >= 0 && x1-1 < m_pVolumeGeometry->getGridRowCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1-1, col);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, lengthPerCol-I);
								p.pixelPosterior(iVolumeIndex);
							}
						}

						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridRowCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1, col);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
					}

					// center
					else if (x2 <= 0.5f+T) {
						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridRowCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1, col);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, lengthPerCol);
								p.pixelPosterior(iVolumeIndex);
							}
						}					
					}

					// down
					else  {
						I = (1.5f - T - x2) / (1.0f - 2.0f*T) * lengthPerCol;

						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1, col);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
						if (x1+1 >= 0 && x1+1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1+1, col);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, lengthPerCol-I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
					}
				}
			} // end loop col
	
			// POLICY: RAY POSTERIOR
			p.rayPosterior(iRayIndex);
	
		} // end loop detector
	} // end loop angles

}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK - vector projection geometry
template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectBlock_internal_vector(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 detX, detY, S, T, I, x, y, c, r, update_c, update_r, offset;
	float32 lengthPerRow, lengthPerCol, inv_pixelLengthX, inv_pixelLengthY, invTminSTimesLengthPerRow, invTminSTimesLengthPerCol;
	int iVolumeIndex, iRayIndex, row, col, iAngle, iDetector;

	const SParProjection * proj = 0;
	const CParallelVecProjectionGeometry2D* pVecProjectionGeometry = dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry);

	inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();
	inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();

	int colCount = m_pVolumeGeometry->getGridColCount();
	int rowCount = m_pVolumeGeometry->getGridRowCount();

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			S = fabs(0.5f * (1.0f - proj->fRayY/proj->fRayX));
			T = fabs(0.5f * (1.0f + proj->fRayY/proj->fRayX));
			lengthPerRow = m_pVolumeGeometry->getPixelLengthX() * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / abs(proj->fRayY);
			invTminSTimesLengthPerRow = lengthPerRow / (T - S);
			update_c = -m_pVolumeGeometry->getPixelLengthY() * (proj->fRayX/proj->fRayY) * inv_pixelLengthX;
		} else {
			S = fabs(0.5f * (1.0f - proj->fRayX/proj->fRayY));
			T = fabs(0.5f * (1.0f + proj->fRayX/proj->fRayY));
			lengthPerCol = m_pVolumeGeometry->getPixelLengthY() * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / abs(proj->fRayX);
			invTminSTimesLengthPerCol = lengthPerCol / (T - S);
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
				y = detY + (proj->fRayY/proj->fRayX)*(m_pVolumeGeometry->pixelColToCenterX(0)-detX);
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