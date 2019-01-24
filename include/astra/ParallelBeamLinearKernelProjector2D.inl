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
/* PROJECT BLOCK - vector projection geometry
   
   Kernel limitations: isotropic pixels (PixelLengthX == PixelLengthY)
  
   For each angle/detector pair:
   
   Let D=(Dx,Dy) denote the centre of the detector (point) in volume coordinates, and
   let R=(Rx,Ry) denote the direction of the ray (vector).
   
   For mainly vertical rays (|Rx|<=|Ry|), 
   let E=(Ex,Ey) denote the centre of the most upper left pixel:
      E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
   and let F=(Fx,Fy) denote a vector to the next pixel 
      F = (PixelLengthX, 0)
   
   The intersection of the ray (D+aR) with the centre line of the upper row of pixels (E+bF) is
      { Dx + a*Rx = Ex + b*Fx
      { Dy + a*Ry = Ey + b*Fy
   Solving for (a,b) results in:
      a = (Ey + b*Fy - Dy)/Ry
        = (Ey - Dy)/Ry
      b = (Dx + a*Rx - Ex)/Fx
        = (Dx + (Ey - Dy)*Rx/Ry - Ex)/Fx
  
   Define c as the x-value of the intersection of the ray with the upper row in pixel coordinates. 
      c = b 
  
   The intersection of the ray (D+aR) with the centre line of the second row of pixels (E'+bF) with
      E'=(WindowMinX + PixelLengthX/2, WindowMaxY - 3*PixelLengthY/2)
   expressed in x-value pixel coordinates is
      c' = (Dx + (Ey' - Dy)*Rx/Ry - Ex)/Fx.
   And thus:
      deltac = c' - c = (Dx + (Ey' - Dy)*Rx/Ry - Ex)/Fx - (Dx + (Ey - Dy)*Rx/Ry - Ex)/Fx
                      = [(Ey' - Dy)*Rx/Ry - (Ey - Dy)*Rx/Ry]/Fx
                      = [Ey' - Ey]*(Rx/Ry)/Fx
                      = [Ey' - Ey]*(Rx/Ry)/Fx
                      = -PixelLengthY*(Rx/Ry)/Fx.
   
   Given c on a certain row, its pixel directly on its left (col), and the distance (offset) to it, can be found: 
      col = floor(c)
      offset = c - col
  
   The index of this pixel is
      volumeIndex = row * colCount + col
  
   The projection kernel is defined by
  
                  LengthPerRow
         /|\
        / | \
     __/  |  \__ 0
       p0 p1 p2
  
   And thus
      W_(rayIndex,volIndex) = (1 - offset) * lengthPerRow
      W_(rayIndex,volIndex+1) = offset * lengthPerRow
  
  
   Mainly horizontal rays (|Rx|<=|Ry|) are handled in a similar fashion:
  
      E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
      F = (0, -PixelLengthX)
  
      a = (Ex + b*Fx - Dx)/Rx = (Ex - Dx)/Rx
      b = (Dy + a*Ry - Ey)/Fy = (Dy + (Ex - Dx)*Ry/Rx - Ey)/Fy
      r = b
      deltar = PixelLengthX*(Ry/Rx)/Fy.
      row = floor(r+1/2)
      offset = r - row
      LengthPerCol = pixelLengthY * sqrt(Rx^2+Ry^2) / |Rx|
  
      W_(rayIndex,volIndex) = (1 - offset) * lengthPerCol
      W_(rayIndex,volIndex+colcount) = offset * lengthPerCol
*/
template <typename Policy>
void CParallelBeamLinearKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
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

	// loop angles
	for (int iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		// variables
		float32 Dx, Dy, Ex, Ey, c, r, deltac, deltar, offset;
		float32 RxOverRy, RyOverRx, lengthPerRow, lengthPerCol;
		int iVolumeIndex, iRayIndex, row, col, iDetector;

		const SParProjection * proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		float32 detSize = sqrt(proj->fDetUX * proj->fDetUX + proj->fDetUY * proj->fDetUY);

		const bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			RxOverRy = proj->fRayX/proj->fRayY;
			lengthPerRow = detSize * m_pVolumeGeometry->getPixelLengthX() * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / abs(proj->fRayY);
			deltac = -pixelLengthY * RxOverRy * inv_pixelLengthX;
		} else {
			RyOverRx = proj->fRayY/proj->fRayX;
			lengthPerCol = detSize * m_pVolumeGeometry->getPixelLengthY() * sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX) / abs(proj->fRayX);
			deltar = -pixelLengthX * RyOverRx * inv_pixelLengthY;
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

			bool isin = false;

			// vertically
			if (vertical) {

				// calculate c for row 0
				c = (Dx + (Ey - Dy)*RxOverRy - Ex) * inv_pixelLengthX;

				// loop rows
				for (row = 0; row < rowCount; ++row, c += deltac) {

					col = int(floor(c));
					if (col < -1 || col >= colCount) { if (!isin) continue; else break; }
					offset = c - float32(col);

					iVolumeIndex = row * colCount + col;
					if (col >= 0) { policy_weight(p, iRayIndex, iVolumeIndex, (1.0f - offset) * lengthPerRow); }
					
					iVolumeIndex++;
					if (col + 1 < colCount) { policy_weight(p, iRayIndex, iVolumeIndex, offset * lengthPerRow); }

					isin = true;
				}
			}

			// horizontally
			else {

				// calculate r for col 0
				r = -(Dy + (Ex - Dx)*RyOverRx - Ey) * inv_pixelLengthY;

				// loop columns
				for (col = 0; col < colCount; ++col, r += deltar) {

					row = int(floor(r));
					if (row < -1 || row >= rowCount) { if (!isin) continue; else break; }
					offset = r - float32(row);

					iVolumeIndex = row * colCount + col;
					if (row >= 0) { policy_weight(p, iRayIndex, iVolumeIndex, (1.0f - offset) * lengthPerCol); }

					iVolumeIndex += colCount;
					if (row + 1 < rowCount) { policy_weight(p, iRayIndex, iVolumeIndex, offset * lengthPerCol); }
					
					isin = true;
				}
			}
	
			// POLICY: RAY POSTERIOR
			p.rayPosterior(iRayIndex);
	
		} // end loop detector
	} // end loop angles

	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry))
		delete pVecProjectionGeometry;
}
