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

#include <cmath>

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


// TODO: Remove duplication with cuda version of this
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

SCylConeProjection* genCylConeProjections(unsigned int iProjAngles,
                                          unsigned int iProjU,
                                          unsigned int iProjV,
                                          double fOriginSourceDistance,
                                          double fOriginDetectorDistance,
                                          double fDetUSize,
                                          double fDetVSize,
                                          double fDetRadius,
                                          const float *pfAngles);

void getCylConeAxes(const SCylConeProjection &p, Vec3 &cyla, Vec3 &cylb, Vec3 &cylc, Vec3 &cylaxis);




}

#endif
