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

#ifndef _INC_ASTRA_GEOMETRYUTIL2D
#define _INC_ASTRA_GEOMETRYUTIL2D

namespace astra {

struct SParProjection {
	// the ray direction
	float fRayX, fRayY;

	// the start of the (linear) detector
	float fDetSX, fDetSY;

	// the length of a single detector pixel
	float fDetUX, fDetUY;


	void translate(double dx, double dy) {
		fDetSX += dx;
		fDetSY += dy;
	}
	void scale(double factor) {
		fRayX *= factor;
		fRayY *= factor;
		fDetSX *= factor;
		fDetSY *= factor;
		fDetUX *= factor;
		fDetUY *= factor;
	}
};


struct SFanProjection {
	// the source
	float fSrcX, fSrcY;

	// the start of the (linear) detector
	float fDetSX, fDetSY;

	// the length of a single detector pixel
	float fDetUX, fDetUY;

	void translate(double dx, double dy) {
		fSrcX += dx;
		fSrcY += dy;
		fDetSX += dx;
		fDetSY += dy;
	}
	void scale(double factor) {
		fSrcX *= factor;
		fSrcY *= factor;
		fDetSX *= factor;
		fDetSY *= factor;
		fDetUX *= factor;
		fDetUY *= factor;
	}
};



SParProjection* genParProjections(unsigned int iProjAngles,
                                  unsigned int iProjDets,
                                  double fDetSize,
                                  const float *pfAngles,
                                  const float *pfExtraOffsets);

SFanProjection* genFanProjections(unsigned int iProjAngles,
                                  unsigned int iProjDets,
                                  double fOriginSource, double fOriginDetector,
                                  double fDetSize,
                                  const float *pfAngles);

bool getParParameters(const SParProjection &proj, unsigned int iProjDets, float &fAngle, float &fDetSize, float &fOffset);

bool getFanParameters(const SFanProjection &proj, unsigned int iProjDets, float &fAngle, float &fOriginSource, float &fOriginDetector, float &fDetSize, float &fOffset);


}

#endif
