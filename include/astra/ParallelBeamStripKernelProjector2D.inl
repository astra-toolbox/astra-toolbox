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
void CParallelBeamStripKernelProjector2D::project(Policy& p)
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
void CParallelBeamStripKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
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
void CParallelBeamStripKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
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
// PROJECT BLOCK
template <typename Policy>
void CParallelBeamStripKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	ASTRA_ASSERT(m_bIsInitialized);

	// Some variables
	float32 theta, t;
	int row, col;
	int iAngle;
	int iDetector;
	float32 res;
	float32 PL, PLimitL, PLimitR;
	float32 xL, xR, XLimitL, XLimitR;
	int x1L,x1R;
	float32 x2L, x2R, updateX;
	int iVolumeIndex, iRayIndex;
	
	float32 sin_theta, cos_theta, inv_sin_theta, inv_cos_theta;
	float32 fabs_sin_theta, fabs_cos_theta, fabs_inv_sin_theta, fabs_inv_cos_theta;
	float32 PW, PH, DW, inv_PW, inv_PH;
	float32 S, T, U, V, inv_4T;

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {
		
		// get values
		theta = m_pProjectionGeometry->getProjectionAngle(iAngle);
		bool switch_t = false;
		if (theta >= 7*PIdiv4) theta -= 2*PI;
		if (theta >= 3*PIdiv4) {
			theta -= PI;
			switch_t = true;
		}

		// Precalculate sin, cos, 1/cos
		sin_theta = sin(theta);
		cos_theta = cos(theta);
		inv_cos_theta = 1.0f / cos_theta; 
		inv_sin_theta = 1.0f / sin_theta;

		fabs_sin_theta = (sin_theta < 0.0f) ? -sin_theta : sin_theta;
		fabs_cos_theta = (cos_theta < 0.0f) ? -cos_theta : cos_theta;
		fabs_inv_cos_theta = (inv_cos_theta < 0.0f) ? -inv_cos_theta : inv_cos_theta;
		fabs_inv_sin_theta = (inv_sin_theta < 0.0f) ? -inv_sin_theta : inv_sin_theta;

		// Other precalculations
		PW = m_pVolumeGeometry->getPixelLengthX();
		PH = m_pVolumeGeometry->getPixelLengthY();
		DW = m_pProjectionGeometry->getDetectorWidth();
		inv_PW = 1.0f / PW;
		inv_PH = 1.0f / PH;

		// [-45?,45?] and [135?,225?]
		if (theta < PIdiv4) {

			// Precalculate kernel limits
			S = -0.5f * fabs_sin_theta * fabs_inv_cos_theta;
			T = -S;
			U = 1.0f + S;
			V = 1.0f - S;
			inv_4T = 0.25f / T;

			updateX = sin_theta * inv_cos_theta;

			// loop detectors
			for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
				iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

				// POLICY: RAY PRIOR
				if (!p.rayPrior(iRayIndex)) continue;

				// get t
				t = m_pProjectionGeometry->indexToDetectorOffset(iDetector);
				if (switch_t) t = -t;
			
				// calculate left strip extremes (volume coordinates)
				PL = (t - sin_theta * m_pVolumeGeometry->pixelRowToCenterY(0) - DW*0.5f) * inv_cos_theta;
				PLimitL = PL - 0.5f * fabs_sin_theta * fabs_inv_cos_theta * PH;
				PLimitR = PLimitL + DW * inv_cos_theta + PH * fabs_sin_theta * fabs_inv_cos_theta; 

				// calculate strip extremes (pixel coordinates)
				XLimitL = (PLimitL - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
				XLimitR = (PLimitR - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
				xL = (PL - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
				xR = xL + (DW * inv_cos_theta) * inv_PW;
	
				// for each row
				for (row = 0; row < m_pVolumeGeometry->getGridRowCount(); ++row) {
				
					// get strip extremes in column indices
					x1L = int((XLimitL > 0.0f) ? XLimitL : XLimitL-1.0f);
					x1R = int((XLimitR > 0.0f) ? XLimitR : XLimitR-1.0f);

					// get coords w.r.t leftmost column hit by strip
					x2L = xL - x1L; 
					x2R = xR - x1L;
					
					// update strip extremes for the next row
					XLimitL += updateX; 
					XLimitR += updateX;
					xL += updateX; 
					xR += updateX; 

					// for each affected col
					for (col = x1L; col <= x1R; ++col) {

						if (col < 0 || col >= m_pVolumeGeometry->getGridColCount()) { x2L -= 1.0f; x2R -= 1.0f;	continue; }

						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, col);
						// POLICY: PIXEL PRIOR
						if (!p.pixelPrior(iVolumeIndex)) { x2L -= 1.0f; x2R -= 1.0f; continue; }
						
						// right
						if (x2R >= V)		res = 1.0f;
						else if (x2R > U)	res = x2R - (x2R-U)*(x2R-U)*inv_4T;
						else if (x2R >= T)	res = x2R;
						else if (x2R > S)	res = (x2R-S)*(x2R-S) * inv_4T;
						else				{ x2L -= 1.0f; x2R -= 1.0f;	continue; }
								
						// left
						if (x2L <= S)		{} // - 0.0f
						else if (x2L < T)	res -= (x2L-S)*(x2L-S) * inv_4T;
						else if (x2L <= U)	res -= x2L;
						else if (x2L < V)	res -= x2L - (x2L-U)*(x2L-U)*inv_4T;
						else				{ x2L -= 1.0f; x2R -= 1.0f;	continue; }

						// POLICY: ADD
						p.addWeight(iRayIndex, iVolumeIndex, PW*PH * res);

						// POLICY: PIXEL POSTERIOR
						p.pixelPosterior(iVolumeIndex);

						x2L -= 1.0f;		
						x2R -= 1.0f;

					} // end col loop

				} // end row loop

				// POLICY: RAY POSTERIOR
				p.rayPosterior(iRayIndex);

			}	// end detector loop

		// [45?,135?] and [225?,315?]
		// horizontaly
		} else {

			// Precalculate kernel limits
			S = -0.5f * fabs_cos_theta * fabs_inv_sin_theta;
			T = -S;
			U = 1.0f + S;
			V = 1.0f - S;
			inv_4T = 0.25f / T;

			updateX = cos_theta * inv_sin_theta;

			// loop detectors
			for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
				iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

				// POLICY: RAY PRIOR
				if (!p.rayPrior(iRayIndex)) continue;

				// get t
				t = m_pProjectionGeometry->indexToDetectorOffset(iDetector);
				if (switch_t) t = -t;
			
				// calculate left strip extremes (volume coordinates)
				PL = (t - cos_theta * m_pVolumeGeometry->pixelColToCenterX(0) + DW*0.5f) * inv_sin_theta;
				PLimitL = PL + 0.5f * fabs_cos_theta * fabs_inv_sin_theta * PW;
				PLimitR = PLimitL - DW * inv_sin_theta - PH * fabs_cos_theta * fabs_inv_sin_theta; 

				// calculate strip extremes (pixel coordinates)
				XLimitL = (m_pVolumeGeometry->getWindowMaxY() - PLimitL) * inv_PH;
				XLimitR = (m_pVolumeGeometry->getWindowMaxY() - PLimitR) * inv_PH;
				xL = (m_pVolumeGeometry->getWindowMaxY() - PL) * inv_PH;
				xR = xL + (DW * fabs_inv_sin_theta) * inv_PH;

				// for each col
				for (col = 0; col < m_pVolumeGeometry->getGridColCount(); ++col) {

					// get strip extremes in column indices
					x1L = int((XLimitL > 0.0f) ? XLimitL : XLimitL-1.0f);
					x1R = int((XLimitR > 0.0f) ? XLimitR : XLimitR-1.0f);

					// get coords w.r.t leftmost column hit by strip
					x2L = xL - x1L; 
					x2R = xR - x1L;
					
					// update strip extremes for the next row
					XLimitL += updateX; 
					XLimitR += updateX;
					xL += updateX; 
					xR += updateX; 

					// for each affected col
					for (row = x1L; row <= x1R; ++row) {

						if (row < 0 || row >= m_pVolumeGeometry->getGridRowCount()) { x2L -= 1.0f; x2R -= 1.0f;	continue; }

						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, col);

						// POLICY: PIXEL PRIOR
						if (!p.pixelPrior(iVolumeIndex)) { x2L -= 1.0f; x2R -= 1.0f; continue; }

						// right
						if (x2R >= V)		res = 1.0f;
						else if (x2R > U)	res = x2R - (x2R-U)*(x2R-U)*inv_4T;
						else if (x2R >= T)	res = x2R;
						else if (x2R > S)	res = (x2R-S)*(x2R-S) * inv_4T;
						else				{ x2L -= 1.0f; x2R -= 1.0f;	continue; }
								
						// left
						if (x2L <= S)		{} // - 0.0f
						else if (x2L < T)	res -= (x2L-S)*(x2L-S) * inv_4T;
						else if (x2L <= U)	res -= x2L;
						else if (x2L < V)	res -= x2L - (x2L-U)*(x2L-U)*inv_4T;
						else				{ x2L -= 1.0f; x2R -= 1.0f;	continue; }

						// POLICY: ADD
						p.addWeight(iRayIndex, iVolumeIndex, PW*PH * res);

						// POLICY: PIXEL POSTERIOR
						p.pixelPosterior(iVolumeIndex);

						x2L -= 1.0f;
						x2R -= 1.0f;

					} // end row loop

				} // end col loop

				// POLICY: RAY POSTERIOR
				p.rayPosterior(iRayIndex);

			} // end detector loop


		} // end theta switch

	} // end angle loop
}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK - vector projection geometry
template <typename Policy>
void CParallelBeamStripKernelProjector2D::projectBlock_internal_vector(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 detX, detY, detLX, detLY, detRX, detRY, S, T, update_c, update_r, offsetL, offsetR, invTminS;
	float32 lengthPerRow, lengthPerCol, inv_pixelLengthX, inv_pixelLengthY, pixelArea;
	int iVolumeIndex, iRayIndex, iRayIndexL, iRayIndexR, row, row_top, row_bottom, col, col_left, col_right, iAngle, iDetector, colCount, rowCount;

	const SParProjection * proj = 0;
	const CParallelVecProjectionGeometry2D* pVecProjectionGeometry = dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry);

	inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();
	inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();
	pixelArea = m_pVolumeGeometry->getPixelLengthX() * m_pVolumeGeometry->getPixelLengthY();

	colCount = m_pVolumeGeometry->getGridColCount();
	rowCount = m_pVolumeGeometry->getGridRowCount();

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			S = 0.5f - 0.5f*fabs(proj->fRayX/proj->fRayY);
			T = 0.5f + 0.5f*fabs(proj->fRayX/proj->fRayY);
			update_c = -m_pVolumeGeometry->getPixelLengthY() * (proj->fRayX/proj->fRayY) * inv_pixelLengthX;
			invTminS = 1.0f / (T-S);
		} else {
			S = 0.5f - 0.5f*fabs(proj->fRayY/proj->fRayX);
			T = 0.5f + 0.5f*fabs(proj->fRayY/proj->fRayX);
			update_r = -m_pVolumeGeometry->getPixelLengthX() * (proj->fRayY/proj->fRayX) * inv_pixelLengthY;
			invTminS = 1.0f / (T-S);
		}

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			detLX = proj->fDetSX + (iDetector+0.0f) * proj->fDetUX;
			detLY = proj->fDetSY + (iDetector+0.0f) * proj->fDetUY;
			detRX = detLX + proj->fDetUX;
			detRY = detLY + proj->fDetUY;

			// vertically
			if (vertical) {

				// calculate cL and cR for row 0
				float32 xL = detLX + (proj->fRayX/proj->fRayY)*(m_pVolumeGeometry->pixelRowToCenterY(0)-detLY);
				float32 cL = (xL - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f;
				float32 xR = detRX + (proj->fRayX/proj->fRayY)*(m_pVolumeGeometry->pixelRowToCenterY(0)-detRY);
				float32 cR = (xR - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f;

				if (cR < cL) {
					float32 tmp = cL;
					cL = cR;
					cR = tmp;
				}

				// for each row
				for (row = 0; row < rowCount; ++row, cL += update_c, cR += update_c) {

					col_left = int(cL-0.5f+S);
					col_right = int(cR+1.5-S);

					if (col_left < 0) col_left = 0; 
					if (col_right > colCount-1) col_right = colCount-1; 

					// for each column
					for (col = col_left; col <= col_right; ++col) {

						iVolumeIndex = row * colCount + col;

						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {

							offsetL = cL - float32(col);
							offsetR = cR - float32(col);

							// right ray edge
							float32 res = 0.0f;
							if (T <= offsetR) 		res = 1.0f;
							else if (S < offsetR) 	res = 1.0f - 0.5f*(T-offsetR)*(T-offsetR)*invTminS;
							else if (-S < offsetR) 	res = 0.5f + offsetR;
							else if (-T < offsetR) 	res = 0.5f*(offsetR+T)*(offsetR+T)*invTminS;

							// left ray edge
							if (T <= offsetL) 		res -= 1.0f;
							else if (S < offsetL) 	res -= 1.0f - 0.5f*(T-offsetL)*(T-offsetL)*invTminS;
							else if (-S < offsetL)  res -= 0.5f + offsetL;
							else if (-T < offsetL)  res -= 0.5f*(offsetL+T)*(offsetL+T)*invTminS;


							p.addWeight(iRayIndex, iVolumeIndex, pixelArea*res);
							p.pixelPosterior(iVolumeIndex);
						}
					}
				}
			}

			// horizontally
			else {

				// calculate rL and rR for row 0
				float32 yL = detLY + (proj->fRayY/proj->fRayX)*(m_pVolumeGeometry->pixelColToCenterX(0)-detLX);
				float32 rL = (m_pVolumeGeometry->getWindowMaxY() - yL) * inv_pixelLengthY - 0.5f;
				float32 yR = detRY + (proj->fRayY/proj->fRayX)*(m_pVolumeGeometry->pixelColToCenterX(0)-detRX);
				float32 rR = (m_pVolumeGeometry->getWindowMaxY() - yR) * inv_pixelLengthY - 0.5f;

				if (rR < rL) {
					float32 tmp = rL;
					rL = rR;
					rR = tmp;
				}

				// for each column
				for (col = 0; col < colCount; ++col, rL += update_r, rR += update_r) {

					row_top = int(rL-0.5f+S);
					row_bottom = int(rR+1.5-S);

					if (row_top < 0) row_top = 0; 
					if (row_bottom > rowCount-1) row_bottom = rowCount-1; 

					// for each row
					for (row = row_top; row <= row_bottom; ++row) {

						iVolumeIndex = row * colCount + col;

						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {

							offsetL = rL - float32(row);
							offsetR = rR - float32(row);

							// right ray edge
							float32 res = 0.0f;
							if (T <= offsetR) 		res = 1.0f;
							else if (S < offsetR) 	res = 1.0f - 0.5f*(T-offsetR)*(T-offsetR)*invTminS;
							else if (-S < offsetR) 	res = 0.5f + offsetR;
							else if (-T < offsetR) 	res = 0.5f*(offsetR+T)*(offsetR+T)*invTminS;

							// left ray edge
							if (T <= offsetL) 		res -= 1.0f;
							else if (S < offsetL) 	res -= 1.0f - 0.5f*(T-offsetL)*(T-offsetL)*invTminS;
							else if (-S < offsetL)  res -= 0.5f + offsetL;
							else if (-T < offsetL)  res -= 0.5f*(offsetL+T)*(offsetL+T)*invTminS;


							p.addWeight(iRayIndex, iVolumeIndex, pixelArea*res);
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
