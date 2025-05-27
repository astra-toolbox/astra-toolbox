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

#ifndef _INC_ASTRA_GEOMETRYUTIL3D
#define _INC_ASTRA_GEOMETRYUTIL3D

#include "Globals.h"

#include <cmath>
#include <vector>
#include <variant>

namespace astra {

class CProjectionGeometry3D;
class CParallelProjectionGeometry3D;
class CParallelVecProjectionGeometry3D;
class CConeProjectionGeometry3D;
class CConeVecProjectionGeometry3D;
class CVolumeGeometry3D;


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

struct SCylConeProjection {
	// the source
	double fSrcX, fSrcY, fSrcZ;

	// the center of the (cylinder-section) detector
	double fDetCX, fDetCY, fDetCZ;

	// the U-edge of a detector pixel
	double fDetUX, fDetUY, fDetUZ;

	// the V-edge of a detector pixel
	double fDetVX, fDetVY, fDetVZ;

	// the radius of the cylinder
	double fDetR;


	void translate(double dx, double dy, double dz) {
		fSrcX += dx;
		fSrcY += dy;
		fSrcZ += dz;
		fDetCX += dx;
		fDetCY += dy;
		fDetCZ += dz;
	}

	// NB: no anisotropic scale function
};


struct SDimensions3D {
	unsigned int iVolX;
	unsigned int iVolY;
	unsigned int iVolZ;
	unsigned int iProjAngles;
	unsigned int iProjU; // number of detectors in the U direction
	unsigned int iProjV; // number of detectors in the V direction
};


struct SVolScale3D {
	float fX = 1.0f;
	float fY = 1.0f;
	float fZ = 1.0f;
};


class _AstraExport Geometry3DParameters {
public:
	using variant_t = std::variant<std::monostate, std::vector<SPar3DProjection>, std::vector<SConeProjection>, std::vector<SCylConeProjection> >;

	Geometry3DParameters() {}
	Geometry3DParameters(variant_t && p) : projs(p) { }

	bool isValid() const {
		return !std::holds_alternative<std::monostate>(projs);
	}

	void clear() {
		projs = variant_t{};
	}

	bool isParallel() const {
		return std::holds_alternative<std::vector<SPar3DProjection>>(projs);
	}
	bool isCone() const {
		return std::holds_alternative<std::vector<SConeProjection>>(projs);
	}
	bool isCylCone() const {
		return std::holds_alternative<std::vector<SCylConeProjection>>(projs);
	}


	const SPar3DProjection *getParallel() const {
		if (!std::holds_alternative<std::vector<SPar3DProjection>>(projs))
			return nullptr;

		return &std::get<std::vector<SPar3DProjection>>(projs)[0];
	}

	const SConeProjection *getCone() const {
		if (!std::holds_alternative<std::vector<SConeProjection>>(projs))
			return nullptr;

		return &std::get<std::vector<SConeProjection>>(projs)[0];
	}
	const SCylConeProjection *getCylCone() const {
		if (!std::holds_alternative<std::vector<SCylConeProjection>>(projs))
			return nullptr;

		return &std::get<std::vector<SCylConeProjection>>(projs)[0];
	}


private:
	variant_t projs;
};





struct Vec3 {
	double x;
	double y;
	double z;
	Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }
	Vec3() : x(0.), y(0.), z(0.) { }
	Vec3 operator+(const Vec3 &b) const {
		return Vec3(x + b.x, y + b.y, z + b.z);
	}
	Vec3 operator-(const Vec3 &b) const {
		return Vec3(x - b.x, y - b.y, z - b.z);
	}
	Vec3 operator-() const {
		return Vec3(-x, -y, -z);
	}
	Vec3 operator*(double s) {
		return Vec3(s*x, s*y, s*z);
	}
	double norm() const {
		return sqrt(x*x + y*y + z*z);
	}
	double dot(const Vec3 &b) const {
		return x*b.x + y*b.y + z*b.z;
	}
};

inline double det3x(const Vec3 &b, const Vec3 &c) {
	return (b.y * c.z - b.z * c.y);
}
inline double det3y(const Vec3 &b, const Vec3 &c) {
	return -(b.x * c.z - b.z * c.x);
}

inline double det3z(const Vec3 &b, const Vec3 &c) {
	return (b.x * c.y - b.y * c.x);
}

inline double det3(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
	return a.x * det3x(b,c) + a.y * det3y(b,c) + a.z * det3z(b,c);
}

inline Vec3 cross3(const Vec3 &a, const Vec3 &b) {
	return Vec3(det3x(a,b), det3y(a,b), det3z(a,b));
}






void computeBP_UV_Coeffs(const SPar3DProjection& proj,
                         double &fUX, double &fUY, double &fUZ, double &fUC,
                         double &fVX, double &fVY, double &fVZ, double &fVC);

void computeBP_UV_Coeffs(const SConeProjection& proj,
                         double &fUX, double &fUY, double &fUZ, double &fUC,
                         double &fVX, double &fVY, double &fVZ, double &fVC,
                         double &fDX, double &fDY, double &fDZ, double &fDC);


std::vector<SConeProjection> genConeProjections(unsigned int iProjAngles,
                                    unsigned int iProjU,
                                    unsigned int iProjV,
                                    double fOriginSourceDistance,
                                    double fOriginDetectorDistance,
                                    double fDetUSize,
                                    double fDetVSize,
                                    const float *pfAngles);

std::vector<SPar3DProjection> genPar3DProjections(unsigned int iProjAngles,
                                      unsigned int iProjU,
                                      unsigned int iProjV,
                                      double fDetUSize,
                                      double fDetVSize,
                                      const float *pfAngles);

std::vector<SCylConeProjection> genCylConeProjections(unsigned int iProjAngles,
                                          unsigned int iProjU,
                                          unsigned int iProjV,
                                          double fOriginSourceDistance,
                                          double fOriginDetectorDistance,
                                          double fDetUSize,
                                          double fDetVSize,
                                          double fDetRadius,
                                          const float *pfAngles);

void getCylConeAxes(const SCylConeProjection &p, Vec3 &cyla, Vec3 &cylb, Vec3 &cylc, Vec3 &cylaxis);

CProjectionGeometry3D* getSubProjectionGeometry_U(const CProjectionGeometry3D* pProjGeom, int u, int size);
CProjectionGeometry3D* getSubProjectionGeometry_V(const CProjectionGeometry3D* pProjGeom, int v, int size);
CProjectionGeometry3D* getSubProjectionGeometry_Angle(const CProjectionGeometry3D* pProjGeom, int th, int size);



bool convertAstraGeometry_dims(const CVolumeGeometry3D* pVolGeom,
                               const CProjectionGeometry3D* pProjGeom,
                               SDimensions3D& dims);

Geometry3DParameters convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                                        const CProjectionGeometry3D* pProjGeom,
                                        SVolScale3D& scale);



}

#endif
