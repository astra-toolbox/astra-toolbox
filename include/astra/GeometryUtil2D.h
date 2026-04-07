/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

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

#include "Globals.h"

#include <vector>
#include <variant>

namespace astra {

class CProjectionGeometry2D;
class CVolumeGeometry2D;

struct SDimensions {
	// Width, height of reconstruction volume
	unsigned int iVolWidth;
	unsigned int iVolHeight;

	// Number of projection angles
	unsigned int iProjAngles;

	// Number of detector pixels
	unsigned int iProjDets;
};


struct SParProjection {
	// the ray direction
	float fRayX, fRayY;

	// the start of the (linear) detector
	float fDetSX, fDetSY;

	// the length of a single detector pixel
	float fDetUX, fDetUY;


	void translate(double dx, double dy) {
		fDetSX = (float)(fDetSX + dx);
		fDetSY = (float)(fDetSY + dy);
	}
	void scale(double factor) {
		fRayX = (float)(fRayX * factor);
		fRayY = (float)(fRayY * factor);
		fDetSX = (float)(fDetSX * factor);
		fDetSY = (float)(fDetSY * factor);
		fDetUX = (float)(fDetUX * factor);
		fDetUY = (float)(fDetUY * factor);
	}

	bool operator==(const SParProjection& o) const {
		return fRayX == o.fRayX && fRayY == o.fRayY &&
		       fDetSX == o.fDetSX && fDetSY == o.fDetSY &&
		       fDetUX == o.fDetUX && fDetUY == o.fDetUY;
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
		fSrcX = (float)(fSrcX + dx);
		fSrcY = (float)(fSrcY + dy);
		fDetSX = (float)(fDetSX + dx);
		fDetSY = (float)(fDetSY + dy);
	}
	void scale(double factor) {
		fSrcX = (float)(fSrcX * factor);
		fSrcY = (float)(fSrcY * factor);
		fDetSX = (float)(fDetSX * factor);
		fDetSY = (float)(fDetSY * factor);
		fDetUX = (float)(fDetUX * factor);
		fDetUY = (float)(fDetUY * factor);
	}

	bool operator==(const SFanProjection& o) const {
		return fSrcX == o.fSrcX && fSrcY == o.fSrcY &&
		       fDetSX == o.fDetSX && fDetSY == o.fDetSY &&
		       fDetUX == o.fDetUX && fDetUY == o.fDetUY;
	}
};

class _AstraExport Geometry2DParameters {
public:
	using variant_t = std::variant<std::monostate, std::vector<SParProjection>, std::vector<SFanProjection> >;

	Geometry2DParameters() { }
	Geometry2DParameters(variant_t && p, SDimensions d, float sc) : projs(std::move(p)), dims(d), fOutputScale(sc) { }

	bool isValid() const {
		return !std::holds_alternative<std::monostate>(projs);
	}

	bool isParallel() const {
		return std::holds_alternative<std::vector<SParProjection>>(projs);
	}
	bool isFan() const {
		return std::holds_alternative<std::vector<SFanProjection>>(projs);
	}

	const SParProjection *getParallel() const {
		if (!std::holds_alternative<std::vector<SParProjection>>(projs))
			return nullptr;

		return &std::get<std::vector<SParProjection>>(projs)[0];
	}

	const SFanProjection *getFan() const {
		if (!std::holds_alternative<std::vector<SFanProjection>>(projs))
			return nullptr;

		return &std::get<std::vector<SFanProjection>>(projs)[0];
	}

	const SDimensions& getDims() const {
		return dims;
	}

	float getOutputScale() const {
		return fOutputScale;
	}

private:
	variant_t projs;

	SDimensions dims;
	float fOutputScale;

};




std::vector<SParProjection> genParProjections(unsigned int iProjAngles,
                                  unsigned int iProjDets,
                                  double fDetSize,
                                  const float *pfAngles,
                                  const float *pfExtraOffsets);

std::vector<SFanProjection> genFanProjections(unsigned int iProjAngles,
                                  unsigned int iProjDets,
                                  double fOriginSource, double fOriginDetector,
                                  double fDetSize,
                                  const float *pfAngles);

bool getParParameters(const SParProjection &proj, unsigned int iProjDets, float &fAngle, float &fDetSize, float &fOffset);

bool getFanParameters(const SFanProjection &proj, unsigned int iProjDets, float &fAngle, float &fOriginSource, float &fOriginDetector, float &fDetSize, float &fOffset);

Geometry2DParameters convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                                          const CProjectionGeometry2D* pProjGeom);


}

#endif
