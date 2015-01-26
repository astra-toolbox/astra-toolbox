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
// PROJECT BLOCK
template <typename Policy>
void CParallelBeamLinearKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
	// variables
	float32 theta, sin_theta, cos_theta, inv_sin_theta, inv_cos_theta, t;
	float32 lengthPerRow, updatePerRow, inv_pixelLengthX;
	float32 lengthPerCol, updatePerCol, inv_pixelLengthY;
	bool switch_t;
	int iAngle, iDetector, iVolumeIndex, iRayIndex;
	int row, col, x1;
	float32 P,x,x2;

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
		inv_cos_theta = 1.0f / cos_theta; 
		inv_sin_theta = 1.0f / sin_theta; 

		// precalculate kernel limits
		lengthPerRow = m_pVolumeGeometry->getPixelLengthY() * inv_cos_theta;
		updatePerRow = sin_theta * inv_cos_theta;
		inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();

		// precalculate kernel limits
		lengthPerCol = m_pVolumeGeometry->getPixelLengthX() * inv_sin_theta;
		updatePerCol = cos_theta * inv_sin_theta;
		inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			// get t
			t = m_pProjectionGeometry->indexToDetectorOffset(iDetector);
			if (switch_t) {
				t = -t;
			}

			// vertically
			if (theta <= PIdiv4) {
			
				// calculate x for row 0
				P = (t - sin_theta * m_pVolumeGeometry->pixelRowToCenterY(0)) * inv_cos_theta;
				x = m_pVolumeGeometry->coordXToColF(P) - 0.5f;

				// for each row
				for (row = 0; row < m_pVolumeGeometry->getGridRowCount(); ++row) {
					
					// get coords
					x1 = int((x > 0.0f) ? x : x-1.0f);
					x2 = x - x1; 
					x += updatePerRow;

					// add weights
					if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridColCount()) {
						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1);
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, (1.0f - x2) * lengthPerRow);
							p.pixelPosterior(iVolumeIndex);
						}
					}
					if (x1+1 >= 0 && x1+1 < m_pVolumeGeometry->getGridColCount()) {
						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, x1+1);
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, (x2) * lengthPerRow);
							p.pixelPosterior(iVolumeIndex);
						}
					}
				}
			}

			// horizontally
			else if (PIdiv4 <= theta && theta <= 3*PIdiv4) {

				// calculate point P
				P = (t - cos_theta * m_pVolumeGeometry->pixelColToCenterX(0)) * inv_sin_theta;
				x = m_pVolumeGeometry->coordYToRowF(P) - 0.5f;

				// for each row
				for (col = 0; col < m_pVolumeGeometry->getGridColCount(); ++col) {

					// get coords
					x1 = int((x > 0.0f) ? x : x-1.0f);
					x2 = x - x1; 
					x += updatePerCol;

					// add weights
					if (x1 >= 0 && x1 < m_pVolumeGeometry->getGridRowCount()) {
						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1, col);
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, (1.0f - x2) * lengthPerCol);
							p.pixelPosterior(iVolumeIndex);		
						}
					}
					if (x1+1 >= 0 && x1+1 < m_pVolumeGeometry->getGridRowCount()) {
						iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(x1+1, col);
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							p.addWeight(iRayIndex, iVolumeIndex, x2 * lengthPerCol);
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

