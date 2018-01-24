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

#ifndef _INC_ASTRA_GEOMETRYUTIL3D
#define _INC_ASTRA_GEOMETRYUTIL3D

namespace astra {

struct SConeProjection {
	// the source
	double fSrcX, fSrcY, fSrcZ;

	// the origin ("bottom left") of the (flat-panel) detector
	double fDetSX, fDetSY, fDetSZ;

	// the U-edge of a detector pixel
	double fDetUX, fDetUY, fDetUZ;

	// the V-edge of a detector pixel
	double fDetVX, fDetVY, fDetVZ;




	void translate(double dx, double dy, double dz) {
		fSrcX += dx;
		fSrcY += dy;
		fSrcZ += dz;
		fDetSX += dx;
		fDetSY += dy;
		fDetSZ += dz;

	}
	void scale(double fx, double fy, double fz) {
		fSrcX *= fx;
		fSrcY *= fy;
		fSrcZ *= fz;
		fDetSX *= fx;
		fDetSY *= fy;
		fDetSZ *= fz;
		fDetUX *= fx;
		fDetUY *= fy;
		fDetUZ *= fz;
		fDetVX *= fx;
		fDetVY *= fy;
		fDetVZ *= fz;
	}
};

struct SPar3DProjection {
	// the ray direction
	double fRayX, fRayY, fRayZ;

	// the origin ("bottom left") of the (flat-panel) detector
	double fDetSX, fDetSY, fDetSZ;

	// the U-edge of a detector pixel
	double fDetUX, fDetUY, fDetUZ;

	// the V-edge of a detector pixel
	double fDetVX, fDetVY, fDetVZ;




	void translate(double dx, double dy, double dz) {
		fDetSX += dx;
		fDetSY += dy;
		fDetSZ += dz;
	}
	void scale(double fx, double fy, double fz) {
		fRayX *= fx;
		fRayY *= fy;
		fRayZ *= fz;
		fDetSX *= fx;
		fDetSY *= fy;
		fDetSZ *= fz;
		fDetUX *= fx;
		fDetUY *= fy;
		fDetUZ *= fz;
		fDetVX *= fx;
		fDetVY *= fy;
		fDetVZ *= fz;
	}
};

void computeBP_UV_Coeffs(const SPar3DProjection& proj,
                         double &fUX, double &fUY, double &fUZ, double &fUC,
                         double &fVX, double &fVY, double &fVZ, double &fVC);

void computeBP_UV_Coeffs(const SConeProjection& proj,
                         double &fUX, double &fUY, double &fUZ, double &fUC,
                         double &fVX, double &fVY, double &fVZ, double &fVC,
                         double &fDX, double &fDY, double &fDZ, double &fDC);


SConeProjection* genConeProjections(unsigned int iProjAngles,
                                    unsigned int iProjU,
                                    unsigned int iProjV,
                                    double fOriginSourceDistance,
                                    double fOriginDetectorDistance,
                                    double fDetUSize,
                                    double fDetVSize,
                                    const float *pfAngles);

SPar3DProjection* genPar3DProjections(unsigned int iProjAngles,
                                      unsigned int iProjU,
                                      unsigned int iProjV,
                                      double fDetUSize,
                                      double fDetVSize,
                                      const float *pfAngles);




}

#endif
