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
void CFanFlatBeamStripKernelProjector2D::project(Policy& p)
{
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CFanFlatBeamStripKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CFanFlatBeamStripKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
	                      _iDetector, _iDetector + 1, p);
}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK
template <typename Policy>
void CFanFlatBeamStripKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	ASTRA_ASSERT(m_bIsInitialized);

	// Some variables
	float32 theta;
	int row, col;
	int iAngle, iDetector;
	float32 res;
	int x1L, x1R;
	float32 x2L, x2R;
	int iVolumeIndex, iRayIndex;
	
	CFanFlatProjectionGeometry2D* projgeom = static_cast<CFanFlatProjectionGeometry2D*>(m_pProjectionGeometry);

	// Other precalculations
	float32 PW = m_pVolumeGeometry->getPixelLengthX();
	float32 PH = m_pVolumeGeometry->getPixelLengthY();
	float32 DW = m_pProjectionGeometry->getDetectorWidth();
	float32 inv_PW = 1.0f / PW;
	float32 inv_PH = 1.0f / PH;

	// calculate alpha's
	float32 alpha;
	float32* cos_alpha = new float32[m_pProjectionGeometry->getDetectorCount() + 1];
	float32* sin_alpha = new float32[m_pProjectionGeometry->getDetectorCount() + 1];
	for (int i = 0; i < m_pProjectionGeometry->getDetectorCount() + 1; ++i) {
		alpha = -atan((i - m_pProjectionGeometry->getDetectorCount()*0.5f) * DW / projgeom->getSourceDetectorDistance());
		cos_alpha[i] = cos(alpha);
		sin_alpha[i] = sin(alpha);
	}

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
		float32 sin_theta = sin(theta);
		float32 cos_theta = cos(theta);

		// [-45?,45?] and [135?,225?]
		if (theta < PIdiv4) {

			// loop detectors
			for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {

				iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

				// POLICY: RAY PRIOR
				if (!p.rayPrior(iRayIndex)) continue;

				float32 sin_theta_left, cos_theta_left;
				float32 sin_theta_right, cos_theta_right;

				// get theta_l = alpha_left + theta and theta_r = alpha_right + theta
				float32 t_l, t_r;
				if (!switch_t) {
					sin_theta_left = sin_theta * cos_alpha[iDetector+1] + cos_theta * sin_alpha[iDetector+1];
					sin_theta_right = sin_theta * cos_alpha[iDetector] + cos_theta * sin_alpha[iDetector];

					cos_theta_left = cos_theta * cos_alpha[iDetector+1] - sin_theta * sin_alpha[iDetector+1];
					cos_theta_right = cos_theta * cos_alpha[iDetector] - sin_theta * sin_alpha[iDetector];

					t_l = sin_alpha[iDetector+1] * projgeom->getOriginSourceDistance();
					t_r = sin_alpha[iDetector] * projgeom->getOriginSourceDistance();

				} else {
					sin_theta_left = sin_theta * cos_alpha[iDetector] + cos_theta * sin_alpha[iDetector];
					sin_theta_right = sin_theta * cos_alpha[iDetector+1] + cos_theta * sin_alpha[iDetector+1];

					cos_theta_left = cos_theta * cos_alpha[iDetector] - sin_theta * sin_alpha[iDetector];
					cos_theta_right = cos_theta * cos_alpha[iDetector+1] - sin_theta * sin_alpha[iDetector+1];

					t_l = -sin_alpha[iDetector] * projgeom->getOriginSourceDistance();
					t_r = -sin_alpha[iDetector+1] * projgeom->getOriginSourceDistance();	
				}

				float32 inv_cos_theta_left = 1.0f / cos_theta_left; 
				float32 inv_cos_theta_right = 1.0f / cos_theta_right; 
	
				float32 updateX_left = sin_theta_left * inv_cos_theta_left;
				float32 updateX_right = sin_theta_right * inv_cos_theta_right;

				// Precalculate kernel limits
				float32 S_l = -0.5f * updateX_left;
				if (S_l > 0) {S_l = -S_l;}
				float32 T_l = -S_l;
				float32 U_l = 1.0f + S_l;
				float32 V_l = 1.0f - S_l;
				float32 inv_4T_l = 0.25f / T_l;

				float32 S_r = -0.5f * updateX_right;
				if (S_r > 0) {S_r = -S_r;}
				float32 T_r = -S_r;
				float32 U_r = 1.0f + S_r;
				float32 V_r = 1.0f - S_r;
				float32 inv_4T_r = 0.25f / T_r;

				// calculate strip extremes (volume coordinates)
				float32 PL = (t_l - sin_theta_left * m_pVolumeGeometry->pixelRowToCenterY(0)) * inv_cos_theta_left;
				float32 PR = (t_r - sin_theta_right * m_pVolumeGeometry->pixelRowToCenterY(0)) * inv_cos_theta_right;
				float32 PLimitL = PL + S_l * PH;
				float32 PLimitR = PR - S_r * PH;
				
				// calculate strip extremes (pixel coordinates)
				float32 XLimitL = (PLimitL - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
				float32 XLimitR = (PLimitR - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
				float32 xL = (PL - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
				float32 xR = (PR - m_pVolumeGeometry->getWindowMinX()) * inv_PW;
	
				// for each row
				for (row = 0; row < m_pVolumeGeometry->getGridRowCount(); ++row) {
				
					// get strip extremes in column indices
					x1L = int((XLimitL > 0.0f) ? XLimitL : XLimitL-1.0f);
					x1R = int((XLimitR > 0.0f) ? XLimitR : XLimitR-1.0f);

					// get coords w.r.t leftmost column hit by strip
					x2L = xL - x1L; 
					x2R = xR - x1L;
					
					// update strip extremes for the next row
					XLimitL += updateX_left; 
					XLimitR += updateX_right;
					xL += updateX_left; 
					xR += updateX_right; 

					// for each affected col
					for (col = x1L; col <= x1R; ++col) {

						if (col < 0 || col >= m_pVolumeGeometry->getGridColCount()) { x2L -= 1.0f; x2R -= 1.0f;	continue; }

						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, col);
						// POLICY: PIXEL PRIOR
						if (!p.pixelPrior(iVolumeIndex)) { x2L -= 1.0f; x2R -= 1.0f; continue; }
						
						// right
						if (x2R >= V_r)			res = 1.0f;
						else if (x2R > U_r)		res = x2R - (x2R-U_r)*(x2R-U_r)*inv_4T_r;
						else if (x2R >= T_r)	res = x2R;
						else if (x2R > S_r)		res = (x2R-S_r)*(x2R-S_r) * inv_4T_r;
						else					{ x2L -= 1.0f; x2R -= 1.0f;	continue; }
								
						// left
						if (x2L <= S_l)			{}
						else if (x2L < T_l)		res -= (x2L-S_l)*(x2L-S_l) * inv_4T_l;
						else if (x2L <= U_l)	res -= x2L;
						else if (x2L < V_l)		res -= x2L - (x2L-U_l)*(x2L-U_l)*inv_4T_l;
						else					{ x2L -= 1.0f; x2R -= 1.0f;	continue; }

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

			// loop detectors
			for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {

				iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

				// POLICY: RAY PRIOR
				if (!p.rayPrior(iRayIndex)) continue;

				// get theta_l = alpha_left + theta and theta_r = alpha_right + theta
				float32 sin_theta_left, cos_theta_left;
				float32 sin_theta_right, cos_theta_right;
				float32 t_l, t_r;
				if (!switch_t) {
					sin_theta_left = sin_theta * cos_alpha[iDetector] + cos_theta * sin_alpha[iDetector];
					sin_theta_right = sin_theta * cos_alpha[iDetector+1] + cos_theta * sin_alpha[iDetector+1];

					cos_theta_left = cos_theta * cos_alpha[iDetector] - sin_theta * sin_alpha[iDetector];
					cos_theta_right = cos_theta * cos_alpha[iDetector+1] - sin_theta * sin_alpha[iDetector+1];

					t_l = sin_alpha[iDetector] * projgeom->getOriginSourceDistance();
					t_r = sin_alpha[iDetector+1] * projgeom->getOriginSourceDistance();

				} else {
					sin_theta_left = sin_theta * cos_alpha[iDetector+1] + cos_theta * sin_alpha[iDetector+1];
					sin_theta_right = sin_theta * cos_alpha[iDetector] + cos_theta * sin_alpha[iDetector];

					cos_theta_left = cos_theta * cos_alpha[iDetector+1] - sin_theta * sin_alpha[iDetector+1];
					cos_theta_right = cos_theta * cos_alpha[iDetector] - sin_theta * sin_alpha[iDetector];

					t_l = -sin_alpha[iDetector+1] * projgeom->getOriginSourceDistance();
					t_r = -sin_alpha[iDetector] * projgeom->getOriginSourceDistance();	
				}

				float32 inv_sin_theta_left = 1.0f / sin_theta_left;
				float32 inv_sin_theta_right = 1.0f / sin_theta_right;

				float32 updateX_left = cos_theta_left * inv_sin_theta_left;
				float32 updateX_right = cos_theta_right * inv_sin_theta_right;

				// Precalculate kernel limits
				float32 S_l = -0.5f * updateX_left;
				if (S_l > 0) { S_l = -S_l; }
				float32 T_l = -S_l;
				float32 U_l = 1.0f + S_l;
				float32 V_l = 1.0f - S_l;
				float32 inv_4T_l = 0.25f / T_l;

				float32 S_r = -0.5f * updateX_right;
				if (S_r > 0) { S_r = -S_r; }
				float32 T_r = -S_r;
				float32 U_r = 1.0f + S_r;
				float32 V_r = 1.0f - S_r;
				float32 inv_4T_r = 0.25f / T_r;

				// calculate strip extremes (volume coordinates)
				float32 PL = (t_l - cos_theta_left * m_pVolumeGeometry->pixelColToCenterX(0)) * inv_sin_theta_left;
				float32 PR = (t_r - cos_theta_right * m_pVolumeGeometry->pixelColToCenterX(0)) * inv_sin_theta_right;
				float32 PLimitL = PL - S_l * PW;
				float32 PLimitR = PR + S_r * PW;
				
				// calculate strip extremes (pixel coordinates)
				float32 XLimitL = (m_pVolumeGeometry->getWindowMaxY() - PLimitL) * inv_PH;
				float32 XLimitR = (m_pVolumeGeometry->getWindowMaxY() - PLimitR) * inv_PH;
				float32 xL = (m_pVolumeGeometry->getWindowMaxY() - PL) * inv_PH;
				float32 xR = (m_pVolumeGeometry->getWindowMaxY() - PR) * inv_PH;

				// for each col
				for (col = 0; col < m_pVolumeGeometry->getGridColCount(); ++col) {

					// get strip extremes in column indices
					x1L = int((XLimitL > 0.0f) ? XLimitL : XLimitL-1.0f);
					x1R = int((XLimitR > 0.0f) ? XLimitR : XLimitR-1.0f);

					// get coords w.r.t leftmost column hit by strip
					x2L = xL - x1L; 
					x2R = xR - x1L;
					
					// update strip extremes for the next row
					XLimitL += updateX_left; 
					XLimitR += updateX_right;
					xL += updateX_left; 
					xR += updateX_right; 

					// for each affected row
					for (row = x1L; row <= x1R; ++row) {

						if (row < 0 || row >= m_pVolumeGeometry->getGridRowCount()) { x2L -= 1.0f; x2R -= 1.0f;	continue; }

						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, col);

						// POLICY: PIXEL PRIOR
						if (!p.pixelPrior(iVolumeIndex)) { x2L -= 1.0f; x2R -= 1.0f; continue; }

						// right
						if (x2R >= V_r)			res = 1.0f;
						else if (x2R > U_r)		res = x2R - (x2R-U_r)*(x2R-U_r)*inv_4T_r;
						else if (x2R >= T_r)	res = x2R;
						else if (x2R > S_r)		res = (x2R-S_r)*(x2R-S_r) * inv_4T_r;
						else					{ x2L -= 1.0f; x2R -= 1.0f;	continue; }
								
						// left
						if (x2L <= S_l)			{}
						else if (x2L < T_l)		res -= (x2L-S_l)*(x2L-S_l) * inv_4T_l;
						else if (x2L <= U_l)	res -= x2L;
						else if (x2L < V_l)		res -= x2L - (x2L-U_l)*(x2L-U_l)*inv_4T_l;
						else					{ x2L -= 1.0f; x2R -= 1.0f;	continue; }

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

		} // end theta switch

	} // end angle loop

	delete[] cos_alpha;
	delete[] sin_alpha;
}

