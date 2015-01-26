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



//----------------------------------------------------------------------------------------
// PROJECT ALL 
template <typename Policy>
void CParallelBeamBlobKernelProjector2D::project(Policy& p)
{
	for (int iAngle = 0; iAngle < m_pProjectionGeometry->getProjectionAngleCount(); ++iAngle) {
		for (int iDetector = 0; iDetector < m_pProjectionGeometry->getDetectorCount(); ++iDetector) {
			projectSingleRay(iAngle, iDetector, p);
		}
	}
}


//----------------------------------------------------------------------------------------
// PROJECT SINGLE PROJECTION
template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	for (int iDetector = 0; iDetector < m_pProjectionGeometry->getDetectorCount(); ++iDetector) {
		projectSingleRay(_iProjection, iDetector, p);
	}
}



//----------------------------------------------------------------------------------------
// PROJECT SINGLE RAY
template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	ASTRA_ASSERT(m_bIsInitialized);

	int iRayIndex = _iProjection * m_pProjectionGeometry->getDetectorCount() + _iDetector;

	// POLICY: RAY PRIOR
	if (!p.rayPrior(iRayIndex)) return;

	// get values
	float32 t = m_pProjectionGeometry->indexToDetectorOffset(_iDetector);
	float32 theta = m_pProjectionGeometry->getProjectionAngle(_iProjection);
	if (theta >= 7*PIdiv4) theta -= 2*PI;

	bool flip = false;

	if (theta >= 3*PIdiv4) {
		theta -= PI;
		t = -t;
		flip = true;
	}


	if (theta <= PIdiv4) { // -pi/4 <= theta <= pi/4

		// precalculate sin, cos, 1/cos
		float32 sin_theta = sin(theta);
		float32 cos_theta = cos(theta);
		float32 inv_cos_theta = 1.0f / cos_theta; 

		// precalculate other stuff
		float32 lengthPerRow = m_pVolumeGeometry->getPixelLengthY() * inv_cos_theta;
		float32 updatePerRow = sin_theta * lengthPerRow;
		float32 inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();
		float32 pixelLengthX_over_blobSize = m_pVolumeGeometry->getPixelLengthX() / m_fBlobSize;
		
		// some variables
		int row, col, xmin, xmax;
		float32 P, x, d;

		// calculate P and x for row 0
		P = (t - sin_theta * m_pVolumeGeometry->pixelRowToCenterY(0)) * inv_cos_theta;
		x = (P - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f;

		// for each row
		for (row = 0; row < m_pVolumeGeometry->getGridRowCount(); ++row) {
			
			// calculate extent
			xmin = (int)ceil((P - m_fBlobSize - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f);
			xmax = (int)floor((P + m_fBlobSize - m_pVolumeGeometry->getWindowMinX()) * inv_pixelLengthX - 0.5f);
	
			// add pixels
			for (col = xmin; col <= xmax; col++) {
				if (col >= 0 && col < m_pVolumeGeometry->getGridColCount()) {
					//d = abs(x - col) * pixelLengthX_over_blobSize;
					//index = (int)(d*m_iBlobSampleCount+0.5f);
					//float32 fWeight = m_pfBlobValues[min(index,m_iBlobSampleCount-1)] * lengthPerRow;

					float32 fWeight;
					int index;
					if ((x >= col) ^ flip) {
						d = abs(x - col) * pixelLengthX_over_blobSize * cos_theta;
						index = (int)(d*m_iBlobSampleCount+0.5f);
						fWeight = m_pfBlobValues[min(index,m_iBlobSampleCount-1)];
					} else {
						d = abs(x - col) * pixelLengthX_over_blobSize * cos_theta;
						index = (int)(d*m_iBlobSampleCount+0.5f);
						fWeight = m_pfBlobValuesNeg[min(index,m_iBlobSampleCount-1)];
					}

					int iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, col);
					// POLICY: PIXEL PRIOR + ADD + POSTERIOR
					if (p.pixelPrior(iVolumeIndex)) {
						p.addWeight(iRayIndex, iVolumeIndex, fWeight);
						p.pixelPosterior(iVolumeIndex);
					}
				}
			}

			// update P and x
			P += updatePerRow;
			x += updatePerRow * inv_pixelLengthX;
		}

	} else { // pi/4 < theta < 3pi/4

		// precalculate sin cos
		float32 sin_90_theta = sin(PIdiv2-theta);
		float32 cos_90_theta = cos(PIdiv2-theta);
		float32 inv_cos_90_theta = 1.0f / cos_90_theta; 

		// precalculate other stuff
		float32 lengthPerCol = m_pVolumeGeometry->getPixelLengthX() * inv_cos_90_theta;
		float32 updatePerCol = sin_90_theta * lengthPerCol;
		float32 inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();
		float32 pixelLengthY_over_blobSize = m_pVolumeGeometry->getPixelLengthY() / m_fBlobSize;

		// some variables
		int row, col, xmin, xmax;
		float32 P,x, d;

		// calculate P and x for col 0
		P = (sin_90_theta * m_pVolumeGeometry->pixelColToCenterX(0) - t) * inv_cos_90_theta;
		x = (P - m_pVolumeGeometry->getWindowMinY()) * inv_pixelLengthY - 0.5f;

		// for each col
		for (col = 0; col < m_pVolumeGeometry->getGridColCount(); ++col) {

			// calculate extent
			xmin = (int)ceil((P - m_fBlobSize - m_pVolumeGeometry->getWindowMinY()) * inv_pixelLengthY - 0.5f);
			xmax = (int)floor((P + m_fBlobSize - m_pVolumeGeometry->getWindowMinY()) * inv_pixelLengthY - 0.5f);

			// add pixels
			for (row = xmin; row <= xmax; row++) {
				if (row >= 0 && row < m_pVolumeGeometry->getGridRowCount()) {
					//d = abs(x - row) * pixelLengthY_over_blobSize;
					//int index = (int)(d*m_iBlobSampleCount+0.5f);
					//float32 fWeight = m_pfBlobValues[min(index,m_iBlobSampleCount-1)] * lengthPerCol;

					float32 fWeight;
					int index;
					if ((x <= row) ^ flip) {
						d = abs(x - row) * pixelLengthY_over_blobSize * cos_90_theta;
						index = (int)(d*m_iBlobSampleCount+0.5f);
						fWeight = m_pfBlobValues[min(index,m_iBlobSampleCount-1)];
					} else {
						d = abs(x - row) * pixelLengthY_over_blobSize * cos_90_theta;
						index = (int)(d*m_iBlobSampleCount+0.5f);
						fWeight = m_pfBlobValuesNeg[min(index,m_iBlobSampleCount-1)];
					}


					int iVolumeIndex = m_pVolumeGeometry->pixelRowColToIndex(row, col);
					// POLICY: PIXEL PRIOR + ADD + POSTERIOR
					if (p.pixelPrior(iVolumeIndex)) {
						p.addWeight(iRayIndex, iVolumeIndex, fWeight);
						p.pixelPosterior(iVolumeIndex);
					}
				}
			}

			// update P and x
			P += updatePerCol;
			x += updatePerCol * inv_pixelLengthY;
		}

	}

	// POLICY: RAY POSTERIOR
	p.rayPosterior(iRayIndex);



}
