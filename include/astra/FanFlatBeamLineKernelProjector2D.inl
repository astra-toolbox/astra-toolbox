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
// PROJECT BLOCK
template <typename Policy>
void CFanFlatBeamLineKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 sin_theta, cos_theta, inv_sin_theta, inv_cos_theta, S, T, t, I, P, x, x2;
	float32 lengthPerRow, updatePerRow, inv_pixelLengthX, lengthPerCol, updatePerCol, inv_pixelLengthY;
	int iVolumeIndex, iRayIndex, row, col, iAngle, iDetector, x1;
	bool switch_t;

	const CFanFlatProjectionGeometry2D* pProjectionGeometry = dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry);
	const CFanFlatVecProjectionGeometry2D* pVecProjectionGeometry = dynamic_cast<CFanFlatVecProjectionGeometry2D*>(m_pProjectionGeometry);

	float32 old_theta, theta, alpha;
	const SFanProjection * proj = 0;

	// loop angles
	for (iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		// get theta
		if (pProjectionGeometry) {
			old_theta = pProjectionGeometry->getProjectionAngle(iAngle);
		}
		else if (pVecProjectionGeometry) {
			proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];
			old_theta = atan2(-proj->fSrcX, proj->fSrcY);
			if (old_theta < 0) old_theta += 2*PI;
		} else {
			assert(false);
		}

		switch_t = false;
		if (old_theta >= 7*PIdiv4) old_theta -= 2*PI;
		if (old_theta >= 3*PIdiv4) {
			old_theta -= PI;
			switch_t = true;
		}

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;

			// get values
			if (pProjectionGeometry) {
				t = -pProjectionGeometry->indexToDetectorOffset(iDetector);
				alpha = atan(t / pProjectionGeometry->getSourceDetectorDistance());
				t = sin(alpha) * pProjectionGeometry->getOriginSourceDistance();
			}
			else if (pVecProjectionGeometry) {
				float32 detX = proj->fDetSX + proj->fDetUX*(0.5f + iDetector);
				float32 detY = proj->fDetSY + proj->fDetUY*(0.5f + iDetector);
				alpha = angleBetweenVectors(-proj->fSrcX, -proj->fSrcY, detX - proj->fSrcX, detY - proj->fSrcY);
				t = sin(alpha) * sqrt(proj->fSrcX*proj->fSrcX + proj->fSrcY*proj->fSrcY);
			} else {
				assert(false);
			}

			if (switch_t) t = -t;
			theta = old_theta + alpha;

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

			// vertically
			if (old_theta <= PIdiv4) {
			
				// calculate x for row 0
				P = (t - sin_theta * m_pVolumeGeometry->pixelRowToCenterY(0)) * inv_cos_theta;
				x = (P - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX;

				// for each row
				for (row = 0; row < m_pVolumeGeometry->getGridRowCount(); ++row) {
					
					// get coords
					x1 = int((x > 0.0f) ? x : x-1.0f);
					x2 = x - x1; 
					x += updatePerRow;

					if (x1 < -1 || x1 > m_pVolumeGeometry->getGridColCount()) continue;

					// left
					if (x2 < 0.5f-S) {
						I = (0.5f - S + x2) / (1.0f - 2.0f*S) * lengthPerRow;

						if (x1-1 >= 0 /*&& x1-1 < m_pVolumeGeometry->getGridColCount()*/) {//x1 is always less than or equal to gridColCount because of the "continue" in the beginning of the for-loop
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
					else  if (x2 <= 1.0f) {
						I = (1.5f - S - x2) / (1.0f - 2.0f*S) * lengthPerRow;

						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridColCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
						if (/*x1+1 >= 0 &&*/ x1+1 < m_pVolumeGeometry->getGridColCount()) {//x1 is always greater than or equal to -1 because of the "continue" in the beginning of the for-loop
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
			//else if (PIdiv4 <= old_theta && old_theta <= 3*PIdiv4) {
			else {

				// calculate point P
				P = (t - cos_theta * m_pVolumeGeometry->pixelColToCenterX(0)) * inv_sin_theta;
				x = (m_pVolumeGeometry->getWindowMaxY() - P) * inv_pixelLengthY;

				// for each col
				for (col = 0; col < m_pVolumeGeometry->getGridColCount(); ++col) {
				
					// get coords
					x1 = int((x > 0.0f) ? x : x-1.0f);
					x2 = x - x1; 
					x += updatePerCol;

					if (x1 < -1 || x1 > m_pVolumeGeometry->getGridRowCount()) continue;

					// up
					if (x2 < 0.5f-T) {
						I = (0.5f - T + x2) / (1.0f - 2.0f*T) * lengthPerCol;

						if (x1-1 >= 0 /*&& x1-1 < m_pVolumeGeometry->getGridRowCount()*/) {//x1 is always less than or equal to gridRowCount because of the "continue" in the beginning of the for-loop
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
					else  if (x2 <= 1.0f) {
						I = (1.5f - T - x2) / (1.0f - 2.0f*T) * lengthPerCol;

						if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridRowCount()) {
							iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1, col);
							// POLICY: PIXEL PRIOR + ADD + POSTERIOR
							if (p.pixelPrior(iVolumeIndex)) {
								p.addWeight(iRayIndex, iVolumeIndex, I);
								p.pixelPosterior(iVolumeIndex);
							}
						}
						if (/*x1+1 >= 0 &&*/ x1+1 < m_pVolumeGeometry->getGridRowCount()) {//x1 is always greater than or equal to -1 because of the "continue" in the beginning of the for-loop
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
