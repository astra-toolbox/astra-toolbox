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
/* PROJECT BLOCK
   
   Kernel limitations: isotropic pixels (PixelLengthX == PixelLengthY)
  
   For each angle/detector pair:
   
   Let DL=(DLx,DLy) denote the left of the detector (point) in volume coordinates, and
   Let DR=(DRx,DRy) denote the right of the detector (point) in volume coordinates, and
   let R=(Rx,Ry) denote the direction of the ray (vector).
   
   For mainly vertical rays (|Rx|<=|Ry|), 
   let E=(Ex,Ey) denote the centre of the most upper left pixel:
      E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
   and let F=(Fx,Fy) denote a vector to the next pixel 
      F = (PixelLengthX, 0)
   
   The intersection of the left edge of the strip (DL+aR) with the centre line of the upper row of pixels (E+bF) is
      { DLx + a*Rx = Ex + b*Fx
      { DLy + a*Ry = Ey + b*Fy
   Solving for (a,b) results in:
      a = (Ey + b*Fy - DLy)/Ry
        = (Ey - DLy)/Ry
      b = (DLx + a*Rx - Ex)/Fx
        = (DLx + (Ey - DLy)*Rx/Ry - Ex)/Fx
  
   Define cL as the x-value of the intersection of the left edge of the strip with the upper row in pixel coordinates. 
      cL = b 
  
   cR, the x-value of the intersection of the right edge of the strip with the upper row in pixel coordinates can be found similarly.
  
   The intersection of the ray (DL+aR) with the left line of the second row of pixels (E'+bF) with
      E'=(WindowMinX + PixelLengthX/2, WindowMaxY - 3*PixelLengthY/2)
   expressed in x-value pixel coordinates is
      cL' = (DLx + (Ey' - DLy)*Rx/Ry - Ex)/Fx.
   And thus:
      deltac = cL' - cL = (DLx + (Ey' - DLy)*Rx/Ry - Ex)/Fx - (DLx + (Ey - DLy)*Rx/Ry - Ex)/Fx
                        = [(Ey' - DLy)*Rx/Ry - (Ey - DLy)*Rx/Ry]/Fx
                        = [Ey' - Ey]*(Rx/Ry)/Fx
                        = [Ey' - Ey]*(Rx/Ry)/Fx
                        = -PixelLengthY*(Rx/Ry)/Fx.
  
   The projection weight for a certain pixel is defined by the area between two points of 
  
           _____       LengthPerRow
         /|  |  |\
        / |  |  | \
     __/  |  |  |  \__ 0
      -T -S  0  S  T
   with S = 1/2 - 1/2*|Rx/Ry|, T = 1/2 + 1/2*|Rx/Ry|, and LengthPerRow = pixelLengthX * sqrt(Rx^2+Ry^2) / |Ry|
   
   For a certain row, all columns that are 'hit' by this kernel lie in the interval
      (col_left, col_right) = (floor(cL-1/2+S), floor(cR+3/2-S))
  
   The offsets for both is
      (offsetL, offsetR) = (cL - floor(col_left), cR - floor(col_left))
  
   The projection weight is found by the difference between the integrated values of the kernel
           offset <= -T   Kernel = 0
      -T < offset <= -S   Kernel = PixelArea/2*(T+offset)^2/(T-S)
      -S < offset <=  S   Kernel = PixelArea/2 + offset
       S < offset <=  T   Kernel = PixelArea - PixelArea/2*(T-offset)^2/(T-S)
       T <= offset:       Kernel = PixelArea
*/
template <typename Policy>
void CParallelBeamStripKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
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
	const float32 pixelArea = pixelLengthX * pixelLengthY;
	const float32 inv_pixelLengthX = 1.0f / pixelLengthX;
	const float32 inv_pixelLengthY = 1.0f / pixelLengthY;
	const int colCount = m_pVolumeGeometry->getGridColCount();
	const int rowCount = m_pVolumeGeometry->getGridRowCount();
	const int detCount = pVecProjectionGeometry->getDetectorCount();

	// loop angles
	for (int iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {

		// variables
		float32 DLx, DLy, DRx, DRy, Ex, Ey, S, T, deltac, deltar, offsetL, offsetR, invTminS;
		float32 res, RxOverRy, RyOverRx, cL, cR, rL, rR;
		int iVolumeIndex, iRayIndex, iDetector;
		int row, row_top, row_bottom, col, col_left, col_right;

		const SParProjection * proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];

		bool vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
		if (vertical) {
			RxOverRy = proj->fRayX/proj->fRayY;
			deltac = -m_pVolumeGeometry->getPixelLengthY() * RxOverRy * inv_pixelLengthX;
			S = 0.5f - 0.5f*fabs(RxOverRy);
			T = 0.5f + 0.5f*fabs(RxOverRy);
			invTminS = 1.0f / (T-S);
		} else {
			RyOverRx = proj->fRayY/proj->fRayX;
			deltar = -m_pVolumeGeometry->getPixelLengthX() * RyOverRx * inv_pixelLengthY;
			S = 0.5f - 0.5f*fabs(RyOverRx);
			T = 0.5f + 0.5f*fabs(RyOverRx);
			invTminS = 1.0f / (T-S);
		}

		Ex = m_pVolumeGeometry->getWindowMinX() + pixelLengthX*0.5f;
		Ey = m_pVolumeGeometry->getWindowMaxY() - pixelLengthY*0.5f;

		// loop detectors
		for (iDetector = _iDetFrom; iDetector < _iDetTo; ++iDetector) {
			
			iRayIndex = iAngle * detCount + iDetector;

			// POLICY: RAY PRIOR
			if (!p.rayPrior(iRayIndex)) continue;
	
			DLx = proj->fDetSX + iDetector * proj->fDetUX;
			DLy = proj->fDetSY + iDetector * proj->fDetUY;
			DRx = DLx + proj->fDetUX;
			DRy = DLy + proj->fDetUY;

			// vertically
			if (vertical) {

				// calculate cL and cR for row 0
				cL = (DLx + (Ey - DLy)*RxOverRy - Ex) * inv_pixelLengthX;
				cR = (DRx + (Ey - DRy)*RxOverRy - Ex) * inv_pixelLengthX;

				if (cR < cL) {
					float32 tmp = cL;
					cL = cR;
					cR = tmp;
				}

				// loop rows
				for (row = 0; row < rowCount; ++row, cL += deltac, cR += deltac) {

					col_left = int(cL-0.5f+S);
					col_right = int(cR+1.5-S);

					if (col_left < 0) col_left = 0; 
					if (col_right > colCount-1) col_right = colCount-1; 

					float32 tmp = float32(col_left);
					offsetL = cL - tmp;
					offsetR = cR - tmp;

					// loop columns
					for (col = col_left; col <= col_right; ++col, offsetL -= 1.0f, offsetR -= 1.0f) {

						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {

							// right ray edge
							if (T <= offsetR)       res = 1.0f;
							else if (S < offsetR)   res = 1.0f - 0.5f*(T-offsetR)*(T-offsetR)*invTminS;
							else if (-S < offsetR)  res = 0.5f + offsetR;
							else if (-T < offsetR)  res = 0.5f*(offsetR+T)*(offsetR+T)*invTminS;
							else                    res = 0.0f;

							// left ray edge
							if (T <= offsetL)       res -= 1.0f;
							else if (S < offsetL)   res -= 1.0f - 0.5f*(T-offsetL)*(T-offsetL)*invTminS;
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
				rL = -(DLy + (Ex - DLx)*RyOverRx - Ey) * inv_pixelLengthY;
				rR = -(DRy + (Ex - DRx)*RyOverRx - Ey) * inv_pixelLengthY;

				if (rR < rL) {
					float32 tmp = rL;
					rL = rR;
					rR = tmp;
				}

				// loop columns
				for (col = 0; col < colCount; ++col, rL += deltar, rR += deltar) {

					row_top = int(rL-0.5f+S);
					row_bottom = int(rR+1.5-S);

					if (row_top < 0) row_top = 0; 
					if (row_bottom > rowCount-1) row_bottom = rowCount-1; 

					float32 tmp = float32(row_top);
					offsetL = rL - tmp;
					offsetR = rR - tmp;

					// loop rows
					for (row = row_top; row <= row_bottom; ++row, offsetL -= 1.0f, offsetR -= 1.0f) {

						iVolumeIndex = row * colCount + col;
						// POLICY: PIXEL PRIOR + ADD + POSTERIOR
						if (p.pixelPrior(iVolumeIndex)) {

							// right ray edge
							if (T <= offsetR)       res = 1.0f;
							else if (S < offsetR)   res = 1.0f - 0.5f*(T-offsetR)*(T-offsetR)*invTminS;
							else if (-S < offsetR)  res = 0.5f + offsetR;
							else if (-T < offsetR)  res = 0.5f*(offsetR+T)*(offsetR+T)*invTminS;
							else                    res = 0.0f;

							// left ray edge
							if (T <= offsetL)       res -= 1.0f;
							else if (S < offsetL)   res -= 1.0f - 0.5f*(T-offsetL)*(T-offsetL)*invTminS;
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

	if (dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry))
		delete pVecProjectionGeometry;
}
