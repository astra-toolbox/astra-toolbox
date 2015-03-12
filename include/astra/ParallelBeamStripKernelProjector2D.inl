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
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
		                  0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamStripKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamStripKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK
template <typename Policy>
void CParallelBeamStripKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 detLX, detLY, detRX, detRY, S, T, update_c, update_r, offsetL, offsetR, invTminS;
	float32 inv_pixelLengthX, inv_pixelLengthY, pixelArea, res, fRxOverRy, fRyOverRx;
	int iVolumeIndex, iRayIndex, iAngle, iDetector;
	int row, row_top, row_bottom, col, col_left, col_right, colCount, rowCount, detCount;
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
	pixelArea = m_pVolumeGeometry->getPixelLengthX() * m_pVolumeGeometry->getPixelLengthY();
	colCount = m_pVolumeGeometry->getGridColCount();
	rowCount = m_pVolumeGeometry->getGridRowCount();
	detCount = pVecProjectionGeometry->getDetectorCount();

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			fRxOverRy = proj->fRayX/proj->fRayY;
			update_c = -m_pVolumeGeometry->getPixelLengthY() * fRxOverRy * inv_pixelLengthX;
			S = 0.5f - 0.5f*fabs(fRxOverRy);
			T = 0.5f + 0.5f*fabs(fRxOverRy);
			invTminS = 1.0f / (T-S);
		} else {
			fRyOverRx = proj->fRayY/proj->fRayX;
			update_r = -m_pVolumeGeometry->getPixelLengthX() * fRyOverRx * inv_pixelLengthY;
			S = 0.5f - 0.5f*fabs(fRyOverRx);
			T = 0.5f + 0.5f*fabs(fRyOverRx);
			invTminS = 1.0f / (T-S);
		}

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * detCount + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			detLX = proj->fDetSX + iDetector * proj->fDetUX;
			detLY = proj->fDetSY + iDetector * proj->fDetUY;
			detRX = detLX + proj->fDetUX;
			detRY = detLY + proj->fDetUY;

			// vertically
			if (vertical) {

				// calculate cL and cR for row 0
				float32 xL = detLX + fRxOverRy*(m_pVolumeGeometry->pixelRowToCenterY(0)-detLY);
				float32 cL = (xL - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f;
				float32 xR = detRX + fRxOverRy*(m_pVolumeGeometry->pixelRowToCenterY(0)-detRY);
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

					offsetL = cL - float32(col_left);
					offsetR = cR - float32(col_left);

					// for each column
					for (col = col_left; col <= col_right; ++col, offsetL -= 1.0f, offsetR -= 1.0f) {

						iVolumeIndex = row * colCount + col;

						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {

							// right ray edge
							if (T <= offsetR) 		res = 1.0f;
							else if (S < offsetR) 	res = 1.0f - 0.5f*(T-offsetR)*(T-offsetR)*invTminS;
							else if (-S < offsetR) 	res = 0.5f + offsetR;
							else if (-T < offsetR) 	res = 0.5f*(offsetR+T)*(offsetR+T)*invTminS;
							else 					res = 0.0f;

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
				float32 yL = detLY + fRyOverRx*(m_pVolumeGeometry->pixelColToCenterX(0)-detLX);
				float32 rL = (m_pVolumeGeometry->getWindowMaxY() - yL) * inv_pixelLengthY - 0.5f;
				float32 yR = detRY + fRyOverRx*(m_pVolumeGeometry->pixelColToCenterX(0)-detRX);
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

					offsetL = rL - float32(row_top);
					offsetR = rR - float32(row_top);

					// for each row
					for (row = row_top; row <= row_bottom; ++row, offsetL -= 1.0f, offsetR -= 1.0f) {

						iVolumeIndex = row * colCount + col;

						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {

							// right ray edge
							if (T <= offsetR) 		res = 1.0f;
							else if (S < offsetR) 	res = 1.0f - 0.5f*(T-offsetR)*(T-offsetR)*invTminS;
							else if (-S < offsetR) 	res = 0.5f + offsetR;
							else if (-T < offsetR) 	res = 0.5f*(offsetR+T)*(offsetR+T)*invTminS;
							else 					res = 0.0f;

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
