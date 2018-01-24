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


template <typename Policy>
void CParallelBeamBlobKernelProjector2D::project(Policy& p)
{
	projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
						  0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
						  0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
	projectBlock_internal(_iProjection, _iProjection + 1,
						  _iDetector, _iDetector + 1, p);
}

//----------------------------------------------------------------------------------------
// PROJECT BLOCK - vector projection geometry
// 
// Kernel limitations: isotropic pixels (PixelLengthX == PixelLengthY)
//
// For each angle/detector pair:
// 
// Let D=(Dx,Dy) denote the centre of the detector (point) in volume coordinates, and
// let R=(Rx,Ry) denote the direction of the ray (vector).
// 
// For mainly vertical rays (|Rx|<=|Ry|), 
// let E=(Ex,Ey) denote the centre of the most upper left pixel:
//    E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
// and let F=(Fx,Fy) denote a vector to the next pixel 
//    F = (PixelLengthX, 0)
// 
// The intersection of the ray (D+aR) with the centre line of the upper row of pixels (E+bF) is
//    { Dx + a*Rx = Ex + b*Fx
//    { Dy + a*Ry = Ey + b*Fy
// Solving for (a,b) results in:
//    a = (Ey + b*Fy - Dy)/Ry
//      = (Ey - Dy)/Ry
//    b = (Dx + a*Rx - Ex)/Fx
//      = (Dx + (Ey - Dy)*Rx/Ry - Ex)/Fx
//
// Define c as the x-value of the intersection of the ray with the upper row in pixel coordinates. 
//    c = b 
//
// The intersection of the ray (D+aR) with the centre line of the second row of pixels (E'+bF) with
//    E'=(WindowMinX + PixelLengthX/2, WindowMaxY - 3*PixelLengthY/2)
// expressed in x-value pixel coordinates is
//    c' = (Dx + (Ey' - Dy)*Rx/Ry - Ex)/Fx.
// And thus:
//    deltac = c' - c = (Dx + (Ey' - Dy)*Rx/Ry - Ex)/Fx - (Dx + (Ey - Dy)*Rx/Ry - Ex)/Fx
//                    = [(Ey' - Dy)*Rx/Ry - (Ey - Dy)*Rx/Ry]/Fx
//                    = [Ey' - Ey]*(Rx/Ry)/Fx
//                    = [Ey' - Ey]*(Rx/Ry)/Fx
//                    = -PixelLengthY*(Rx/Ry)/Fx.
// 
// Given c on a certain row, its pixel directly on its left (col), and the distance (offset) to it, can be found: 
//    col = floor(c)
//    offset = c - col
//
// The index of this pixel is
//    volumeIndex = row * colCount + col
//
//
// Mainly horizontal rays (|Rx|<=|Ry|) are handled in a similar fashion:
//
//    E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
//    F = (0, -PixelLengthX)
//
//    a = (Ex + b*Fx - Dx)/Rx = (Ex - Dx)/Rx
//    b = (Dy + a*Ry - Ey)/Fy = (Dy + (Ex - Dx)*Ry/Rx - Ey)/Fy
//    r = b
//    deltar = PixelLengthX*(Ry/Rx)/Fy.
//    row = floor(r+1/2)
//    offset = r - row
//
template <typename Policy>
void CParallelBeamBlobKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
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
	const float32 inv_pixelLengthX = 1.0f / m_pVolumeGeometry->getPixelLengthX();
	const float32 inv_pixelLengthY = 1.0f / m_pVolumeGeometry->getPixelLengthY();
	const int colCount = m_pVolumeGeometry->getGridColCount();
	const int rowCount = m_pVolumeGeometry->getGridRowCount();

	// loop angles
	for (int iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		// variables
		float32 Dx, Dy, Ex, Ey, c, r, deltac, deltar, offset, invBlobExtent, RxOverRy, RyOverRx;
		int iVolumeIndex, iRayIndex, row, col, iDetector;
		int col_left, col_right, row_top, row_bottom, index;

		const SParProjection * proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			RxOverRy = proj->fRayX/proj->fRayY;
			deltac = -m_pVolumeGeometry->getPixelLengthY() * (proj->fRayX/proj->fRayY) * inv_pixelLengthX;
			invBlobExtent = m_pVolumeGeometry->getPixelLengthY() / abs(m_fBlobSize * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / proj->fRayY);
		} else {
			RyOverRx = proj->fRayY/proj->fRayX;
			deltar = -m_pVolumeGeometry->getPixelLengthX() * (proj->fRayY/proj->fRayX) * inv_pixelLengthY;
			invBlobExtent = m_pVolumeGeometry->getPixelLengthX() / abs(m_fBlobSize * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / proj->fRayX);
		}

		Ex = m_pVolumeGeometry->getWindowMinX() + pixelLengthX*0.5f;
		Ey = m_pVolumeGeometry->getWindowMaxY() - pixelLengthY*0.5f;

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * m_pProjectionGeometry->getDetectorCount() + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			Dx = proj->fDetSX + (iDetector+0.5f) * proj->fDetUX;
			Dy = proj->fDetSY + (iDetector+0.5f) * proj->fDetUY;

			// vertically
			if (vertical) {

				// calculate c for row 0
				c = (Dx + (Ey - Dy)*RxOverRy - Ex) * inv_pixelLengthX;

				// loop rows
				for (row = 0; row < rowCount; ++row, c += deltac) {

					col_left = int(c - 0.5f - m_fBlobSize);
					col_right = int(c + 0.5f + m_fBlobSize);

					if (col_left < 0) col_left = 0; 
					if (col_right > colCount-1) col_right = colCount-1; 

					// loop columns
					for (col = col_left; col <= col_right; ++col) {

						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							offset = abs(c - float32(col)) * invBlobExtent;
							index = (int)(offset*m_iBlobSampleCount+0.5f);
							p.addWeight(iRayIndex, iVolumeIndex, m_pfBlobValues[min(index,m_iBlobSampleCount-1)]);
							p.pixelPosterior(iVolumeIndex);
						}
					}
				}
			}

			// horizontally
			else {

				// calculate r for col 0
				r = -(Dy + (Ex - Dx)*RyOverRx - Ey) * inv_pixelLengthY;

				// loop columns
				for (col = 0; col < colCount; ++col, r += deltar) {

					row_top = int(r - 0.5f - m_fBlobSize);
					row_bottom = int(r + 0.5f + m_fBlobSize);

					if (row_top < 0) row_top = 0; 
					if (row_bottom > rowCount-1) row_bottom = rowCount-1; 

					// loop rows
					for (row = row_top; row <= row_bottom; ++row) {

						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {
							offset = abs(r - float32(row)) * invBlobExtent;
							index = (int)(offset*m_iBlobSampleCount+0.5f);
							p.addWeight(iRayIndex, iVolumeIndex, m_pfBlobValues[min(index,m_iBlobSampleCount-1)]);
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
