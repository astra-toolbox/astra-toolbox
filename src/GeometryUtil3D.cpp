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

#include "astra/GeometryUtil3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/CylConeVecProjectionGeometry3D.h"
#include "astra/VolumeGeometry3D.h"


#include <cmath>

namespace astra {


std::vector<SConeProjection> genConeProjections(unsigned int iProjAngles,
                                    unsigned int iProjU,
                                    unsigned int iProjV,
                                    double fOriginSourceDistance,
                                    double fOriginDetectorDistance,
                                    double fDetUSize,
                                    double fDetVSize,
                                    const float *pfAngles)
{
	SConeProjection base;
	base.fSrcX = 0.0f;
	base.fSrcY = -fOriginSourceDistance;
	base.fSrcZ = 0.0f;

	base.fDetSX = iProjU * fDetUSize * -0.5f;
	base.fDetSY = fOriginDetectorDistance;
	base.fDetSZ = iProjV * fDetVSize * -0.5f;

	base.fDetUX = fDetUSize;
	base.fDetUY = 0.0f;
	base.fDetUZ = 0.0f;

	base.fDetVX = 0.0f;
	base.fDetVY = 0.0f;
	base.fDetVZ = fDetVSize;

	std::vector<SConeProjection> p;
	p.resize(iProjAngles);

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); p[i].f##name##Z = base.f##name##Z; } while(0)

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		ROTATE0(Src, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
		ROTATE0(DetV, i, pfAngles[i]);
	}

#undef ROTATE0

	return p;
}

std::vector<SCylConeProjection> genCylConeProjections(unsigned int iProjAngles,
                                          unsigned int iProjU,
                                          unsigned int iProjV,
                                          double fOriginSourceDistance,
                                          double fOriginDetectorDistance,
                                          double fDetUSize,
                                          double fDetVSize,
                                          double fDetRadius,
                                          const float *pfAngles)
{
	SCylConeProjection base;
	base.fSrcX = 0.0f;
	base.fSrcY = -fOriginSourceDistance;
	base.fSrcZ = 0.0f;

	base.fDetCX = 0.0f;
	base.fDetCY = fOriginDetectorDistance;
	base.fDetCZ = 0.0f;

	base.fDetUX = fDetUSize;
	base.fDetUY = 0.0f;
	base.fDetUZ = 0.0f;

	base.fDetVX = 0.0f;
	base.fDetVY = 0.0f;
	base.fDetVZ = fDetVSize;

	base.fDetR = fDetRadius;

	std::vector<SCylConeProjection> p;
	p.resize(iProjAngles);

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); p[i].f##name##Z = base.f##name##Z; } while(0)

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		ROTATE0(Src, i, pfAngles[i]);
		ROTATE0(DetC, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
		ROTATE0(DetV, i, pfAngles[i]);
		p[i].fDetR = base.fDetR;
	}

#undef ROTATE0

	return p;
}


std::vector<SPar3DProjection> genPar3DProjections(unsigned int iProjAngles,
                                      unsigned int iProjU,
                                      unsigned int iProjV,
                                      double fDetUSize,
                                      double fDetVSize,
                                      const float *pfAngles)
{
	SPar3DProjection base;
	base.fRayX = 0.0f;
	base.fRayY = 1.0f;
	base.fRayZ = 0.0f;

	base.fDetSX = iProjU * fDetUSize * -0.5f;
	base.fDetSY = 0.0f;
	base.fDetSZ = iProjV * fDetVSize * -0.5f;

	base.fDetUX = fDetUSize;
	base.fDetUY = 0.0f;
	base.fDetUZ = 0.0f;

	base.fDetVX = 0.0f;
	base.fDetVY = 0.0f;
	base.fDetVZ = fDetVSize;

	std::vector<SPar3DProjection> p;
	p.resize(iProjAngles);

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); p[i].f##name##Z = base.f##name##Z; } while(0)

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		ROTATE0(Ray, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
		ROTATE0(DetV, i, pfAngles[i]);
	}

#undef ROTATE0

	return p;
}


// Utility function to get newly allocated copy of projection vectors for any geometry
template<class V, class P>
static std::vector<V> getProjectionVectors(const P* geom);

template<>
std::vector<SConeProjection> getProjectionVectors(const CConeProjectionGeometry3D* pProjGeom)
{
	return genConeProjections(pProjGeom->getProjectionCount(),
	                          pProjGeom->getDetectorColCount(),
	                          pProjGeom->getDetectorRowCount(),
	                          pProjGeom->getOriginSourceDistance(),
	                          pProjGeom->getOriginDetectorDistance(),
	                          pProjGeom->getDetectorSpacingX(),
	                          pProjGeom->getDetectorSpacingY(),
	                          pProjGeom->getProjectionAngles());
}

template<>
std::vector<SConeProjection> getProjectionVectors(const CConeVecProjectionGeometry3D* pProjGeom)
{
	int nth = pProjGeom->getProjectionCount();

	std::vector<SConeProjection> p;
	p.resize(nth);

	for (int i = 0; i < nth; ++i)
		p[i] = pProjGeom->getProjectionVectors()[i];

	return p;
}

template<>
std::vector<SPar3DProjection> getProjectionVectors(const CParallelProjectionGeometry3D* pProjGeom)
{
	return genPar3DProjections(pProjGeom->getProjectionCount(),
	                           pProjGeom->getDetectorColCount(),
	                           pProjGeom->getDetectorRowCount(),
	                           pProjGeom->getDetectorSpacingX(),
	                           pProjGeom->getDetectorSpacingY(),
	                           pProjGeom->getProjectionAngles());
}

template<>
std::vector<SPar3DProjection> getProjectionVectors(const CParallelVecProjectionGeometry3D* pProjGeom)
{
	int nth = pProjGeom->getProjectionCount();

	std::vector<SPar3DProjection> p;
	p.resize(nth);

	for (int i = 0; i < nth; ++i)
		p[i] = pProjGeom->getProjectionVectors()[i];

	return p;
}

template<>
std::vector<SCylConeProjection> getProjectionVectors(const CCylConeVecProjectionGeometry3D* pProjGeom)
{
	int nth = pProjGeom->getProjectionCount();

	std::vector<SCylConeProjection> p;
	p.resize(nth);

	for (int i = 0; i < nth; ++i)
		p[i] = pProjGeom->getProjectionVectors()[i];

	return p;
}


// Translate detector location along u axis
template<class V>
static void translateDetectorVectorsU(std::vector<V> &projs, double du)
{
	for (auto &p : projs) {
		p.fDetSX += du * p.fDetUX;
		p.fDetSY += du * p.fDetUY;
		p.fDetSZ += du * p.fDetUZ;
	}
}

// Translate detector location along v axis
template<class V>
static void translateDetectorVectorsV(std::vector<V> &projs, double dv)
{
	for (auto &p : projs) {
		p.fDetSX += dv * p.fDetVX;
		p.fDetSY += dv * p.fDetVY;
		p.fDetSZ += dv * p.fDetVZ;
	}
}

template<>
void translateDetectorVectorsV<SCylConeProjection>(std::vector<SCylConeProjection> &projs, double dv)
{
	for (auto &p : projs) {
		p.fDetCX += dv * p.fDetVX;
		p.fDetCY += dv * p.fDetVY;
		p.fDetCZ += dv * p.fDetVZ;
	}
}

void rotateProjectionVectorsU(std::vector<SCylConeProjection> &pProjs, double du)
{
	for (SCylConeProjection& p : pProjs) {
		// TODO: reduce code duplication

		double R = p.fDetR;
		Vec3 u(p.fDetUX, p.fDetUY, p.fDetUZ); // u (tangential) direction
		Vec3 v(p.fDetVX, p.fDetVY, p.fDetVZ); // v (axial) direction
		Vec3 s(p.fSrcX, p.fSrcY, p.fSrcZ);    // source
		Vec3 d(p.fDetCX, p.fDetCY, p.fDetCZ); // center of detector

		double fDetUT = u.norm() / R; // angular increment

		Vec3 cyla = -cross3(u, v) * (R / (u.norm() * v.norm())); // radial direction
		Vec3 cylc = d - cyla;                                    // center of cylinder
		Vec3 cylb = u * (R / u.norm());                          // tangential direction

		double theta = fDetUT * du;

		Vec3 dd = cylc + cyla * cos(theta) + cylb * sin(theta);
		Vec3 uu = (cylb * cos(theta) - cyla * sin(theta)) * (u.norm() / R);

		p.fDetCX = dd.x;
		p.fDetCY = dd.y;
		p.fDetCZ = dd.z;
		p.fDetUX = uu.x;
		p.fDetUY = uu.y;
		p.fDetUZ = uu.z;
	}
}



CProjectionGeometry3D* getSubProjectionGeometry_U(const CProjectionGeometry3D* pProjGeom, int u, int size)
{
	// First convert to vectors, then translate, then convert into new object

	const CConeProjectionGeometry3D* conegeom = dynamic_cast<const CConeProjectionGeometry3D*>(pProjGeom);
	const CParallelProjectionGeometry3D* par3dgeom = dynamic_cast<const CParallelProjectionGeometry3D*>(pProjGeom);
	const CParallelVecProjectionGeometry3D* parvec3dgeom = dynamic_cast<const CParallelVecProjectionGeometry3D*>(pProjGeom);
	const CConeVecProjectionGeometry3D* conevec3dgeom = dynamic_cast<const CConeVecProjectionGeometry3D*>(pProjGeom);
	const CCylConeVecProjectionGeometry3D* cylconevec3dgeom = dynamic_cast<const CCylConeVecProjectionGeometry3D*>(pProjGeom);

	if (conegeom || conevec3dgeom) {
		std::vector<SConeProjection> coneProjs;
		if (conegeom) {
			coneProjs = getProjectionVectors<SConeProjection>(conegeom);
		} else {
			coneProjs = getProjectionVectors<SConeProjection>(conevec3dgeom);
		}

		translateDetectorVectorsU(coneProjs, u);

		CProjectionGeometry3D* ret = new CConeVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                              pProjGeom->getDetectorRowCount(),
		                                                              size,
		                                                              std::move(coneProjs));


		return ret;
	} else if (par3dgeom || parvec3dgeom) {
		std::vector<SPar3DProjection> parProjs;
		if (par3dgeom) {
			parProjs = getProjectionVectors<SPar3DProjection>(par3dgeom);
		} else {
			parProjs = getProjectionVectors<SPar3DProjection>(parvec3dgeom);
		}

		translateDetectorVectorsU(parProjs, u);

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                  pProjGeom->getDetectorRowCount(),
		                                                                  size,
		                                                                  std::move(parProjs));

		return ret;
	} else if (cylconevec3dgeom) {
		std::vector<SCylConeProjection> cylConeProjs = getProjectionVectors<SCylConeProjection>(cylconevec3dgeom);
		// relative position of center
		double du = (u + 0.5 * size) - 0.5 * pProjGeom->getDetectorColCount();

		rotateProjectionVectorsU(cylConeProjs, du);

		CProjectionGeometry3D* ret = new CCylConeVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                 pProjGeom->getDetectorRowCount(),
		                                                                 size,
		                                                                 std::move(cylConeProjs));

		return ret;
	} else {
		assert(false);
		return nullptr;
	}
}



CProjectionGeometry3D* getSubProjectionGeometry_V(const CProjectionGeometry3D* pProjGeom, int v, int size)
{
	// First convert to vectors, then translate, then convert into new object

	const CConeProjectionGeometry3D* conegeom = dynamic_cast<const CConeProjectionGeometry3D*>(pProjGeom);
	const CParallelProjectionGeometry3D* par3dgeom = dynamic_cast<const CParallelProjectionGeometry3D*>(pProjGeom);
	const CParallelVecProjectionGeometry3D* parvec3dgeom = dynamic_cast<const CParallelVecProjectionGeometry3D*>(pProjGeom);
	const CConeVecProjectionGeometry3D* conevec3dgeom = dynamic_cast<const CConeVecProjectionGeometry3D*>(pProjGeom);
	const CCylConeVecProjectionGeometry3D* cylconevec3dgeom = dynamic_cast<const CCylConeVecProjectionGeometry3D*>(pProjGeom);

	if (conegeom || conevec3dgeom) {
		std::vector<SConeProjection> coneProjs;
		if (conegeom) {
			coneProjs = getProjectionVectors<SConeProjection>(conegeom);
		} else {
			coneProjs = getProjectionVectors<SConeProjection>(conevec3dgeom);
		}

		translateDetectorVectorsV(coneProjs, v);

		CProjectionGeometry3D* ret = new CConeVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                              size,
		                                                              pProjGeom->getDetectorColCount(),
		                                                              std::move(coneProjs));


		return ret;
	} else if (par3dgeom || parvec3dgeom) {
		std::vector<SPar3DProjection> parProjs;
		if (par3dgeom) {
			parProjs = getProjectionVectors<SPar3DProjection>(par3dgeom);
		} else {
			parProjs = getProjectionVectors<SPar3DProjection>(parvec3dgeom);
		}

		translateDetectorVectorsV(parProjs, v);

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                  size,
		                                                                  pProjGeom->getDetectorColCount(),
		                                                                  std::move(parProjs));

		return ret;
	} else if (cylconevec3dgeom) {
		std::vector<SCylConeProjection> cylConeProjs = getProjectionVectors<SCylConeProjection>(cylconevec3dgeom);

		double dv = (v + 0.5 * size) - 0.5 * pProjGeom->getDetectorRowCount();

		translateDetectorVectorsV(cylConeProjs, dv);

		CProjectionGeometry3D* ret = new CCylConeVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                 size,
		                                                                 pProjGeom->getDetectorColCount(),
		                                                                 std::move(cylConeProjs));

		return ret;
	} else {
		assert(false);
		return nullptr;
	}

}

CProjectionGeometry3D* getSubProjectionGeometry_Angle(const CProjectionGeometry3D* pProjGeom, int th, int size)
{
	// First convert to vectors, then convert into new object

	const CConeProjectionGeometry3D* conegeom = dynamic_cast<const CConeProjectionGeometry3D*>(pProjGeom);
	const CParallelProjectionGeometry3D* par3dgeom = dynamic_cast<const CParallelProjectionGeometry3D*>(pProjGeom);
	const CParallelVecProjectionGeometry3D* parvec3dgeom = dynamic_cast<const CParallelVecProjectionGeometry3D*>(pProjGeom);
	const CConeVecProjectionGeometry3D* conevec3dgeom = dynamic_cast<const CConeVecProjectionGeometry3D*>(pProjGeom);
	const CCylConeVecProjectionGeometry3D* cylconevec3dgeom = dynamic_cast<const CCylConeVecProjectionGeometry3D*>(pProjGeom);

	if (conegeom || conevec3dgeom) {
		std::vector<SConeProjection> coneProjs;
		if (conegeom) {
			coneProjs = getProjectionVectors<SConeProjection>(conegeom);
		} else {
			coneProjs = getProjectionVectors<SConeProjection>(conevec3dgeom);
		}

		CProjectionGeometry3D* ret = new CConeVecProjectionGeometry3D(size,
		                                                              pProjGeom->getDetectorRowCount(),
		                                                              pProjGeom->getDetectorColCount(),
									      std::vector<SConeProjection>(coneProjs.begin() + th,
										                           coneProjs.begin() + th+size));


		return ret;
	} else if (par3dgeom || parvec3dgeom) {
		std::vector<SPar3DProjection> parProjs;
		if (par3dgeom) {
			parProjs = getProjectionVectors<SPar3DProjection>(par3dgeom);
		} else {
			parProjs = getProjectionVectors<SPar3DProjection>(parvec3dgeom);
		}

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(size,
		                                                                  pProjGeom->getDetectorRowCount(),
		                                                                  pProjGeom->getDetectorColCount(),
		                                                                  std::vector<SPar3DProjection>(parProjs.begin() + th,
		                                                                                                parProjs.begin() + th+size));

		return ret;
	} else if (cylconevec3dgeom) {
		std::vector<SCylConeProjection> cylConeProjs = getProjectionVectors<SCylConeProjection>(cylconevec3dgeom);

		CProjectionGeometry3D* ret = new CCylConeVecProjectionGeometry3D(size,
		                                                                 pProjGeom->getDetectorRowCount(),
		                                                                 pProjGeom->getDetectorColCount(),
		                                                                 std::vector<SCylConeProjection>(cylConeProjs.begin() + th,
		                                                                                                 cylConeProjs.begin() + th+size));

		return ret;
	} else {
		assert(false);
		return nullptr;
	}
}






// (See declaration in header for (mathematical) description of these functions)


void computeBP_UV_Coeffs(const SPar3DProjection& proj, double &fUX, double &fUY, double &fUZ, double &fUC,
                                                       double &fVX, double &fVY, double &fVZ, double &fVC)
{
	double denom = (proj.fRayX*proj.fDetUY*proj.fDetVZ - proj.fRayX*proj.fDetUZ*proj.fDetVY - proj.fRayY*proj.fDetUX*proj.fDetVZ + proj.fRayY*proj.fDetUZ*proj.fDetVX + proj.fRayZ*proj.fDetUX*proj.fDetVY - proj.fRayZ*proj.fDetUY*proj.fDetVX);

	fUX = ( - (proj.fRayY*proj.fDetVZ - proj.fRayZ*proj.fDetVY)) / denom;
	fUY = ( (proj.fRayX*proj.fDetVZ - proj.fRayZ*proj.fDetVX)) / denom;
	fUZ = (- (proj.fRayX*proj.fDetVY - proj.fRayY*proj.fDetVX) ) / denom;
	fUC = (-(proj.fDetSY*proj.fDetVZ - proj.fDetSZ*proj.fDetVY)*proj.fRayX + (proj.fRayY*proj.fDetVZ - proj.fRayZ*proj.fDetVY)*proj.fDetSX - (proj.fRayY*proj.fDetSZ - proj.fRayZ*proj.fDetSY)*proj.fDetVX) / denom;

	fVX = ((proj.fRayY*proj.fDetUZ - proj.fRayZ*proj.fDetUY) ) / denom;
	fVY = (- (proj.fRayX*proj.fDetUZ - proj.fRayZ*proj.fDetUX) ) / denom;
	fVZ = ((proj.fRayX*proj.fDetUY - proj.fRayY*proj.fDetUX) ) / denom;
	fVC = ((proj.fDetSY*proj.fDetUZ - proj.fDetSZ*proj.fDetUY)*proj.fRayX - (proj.fRayY*proj.fDetUZ - proj.fRayZ*proj.fDetUY)*proj.fDetSX + (proj.fRayY*proj.fDetSZ - proj.fRayZ*proj.fDetSY)*proj.fDetUX ) / denom;
}



void computeBP_UV_Coeffs(const SConeProjection& proj, double &fUX, double &fUY, double &fUZ, double &fUC,
                                                      double &fVX, double &fVY, double &fVZ, double &fVC,
                                                      double &fDX, double &fDY, double &fDZ, double &fDC)
{
	fUX = (proj.fDetSZ - proj.fSrcZ)*proj.fDetVY - (proj.fDetSY - proj.fSrcY)*proj.fDetVZ;
	fUY = (proj.fDetSX - proj.fSrcX)*proj.fDetVZ -(proj.fDetSZ - proj.fSrcZ)*proj.fDetVX;
	fUZ = (proj.fDetSY - proj.fSrcY)*proj.fDetVX - (proj.fDetSX - proj.fSrcX)*proj.fDetVY;
	fUC = (proj.fDetSY*proj.fDetVZ - proj.fDetSZ*proj.fDetVY)*proj.fSrcX - (proj.fDetSX*proj.fDetVZ - proj.fDetSZ*proj.fDetVX)*proj.fSrcY + (proj.fDetSX*proj.fDetVY - proj.fDetSY*proj.fDetVX)*proj.fSrcZ;

	fVX = (proj.fDetSY - proj.fSrcY)*proj.fDetUZ-(proj.fDetSZ - proj.fSrcZ)*proj.fDetUY;
	fVY = (proj.fDetSZ - proj.fSrcZ)*proj.fDetUX - (proj.fDetSX - proj.fSrcX)*proj.fDetUZ;
	fVZ = (proj.fDetSX - proj.fSrcX)*proj.fDetUY-(proj.fDetSY - proj.fSrcY)*proj.fDetUX;
	fVC = -(proj.fDetSY*proj.fDetUZ - proj.fDetSZ*proj.fDetUY)*proj.fSrcX + (proj.fDetSX*proj.fDetUZ - proj.fDetSZ*proj.fDetUX)*proj.fSrcY - (proj.fDetSX*proj.fDetUY - proj.fDetSY*proj.fDetUX)*proj.fSrcZ;

	fDX = proj.fDetUY*proj.fDetVZ - proj.fDetUZ*proj.fDetVY;
	fDY = proj.fDetUZ*proj.fDetVX - proj.fDetUX*proj.fDetVZ;
	fDZ = proj.fDetUX*proj.fDetVY - proj.fDetUY*proj.fDetVX;
	fDC = -proj.fSrcX * (proj.fDetUY*proj.fDetVZ - proj.fDetUZ*proj.fDetVY) - proj.fSrcY * (proj.fDetUZ*proj.fDetVX - proj.fDetUX*proj.fDetVZ) - proj.fSrcZ * (proj.fDetUX*proj.fDetVY - proj.fDetUY*proj.fDetVX);
}


void getCylConeAxes(const SCylConeProjection &p, Vec3 &cyla, Vec3 &cylb, Vec3 &cylc, Vec3 &cylaxis)
{
	double R = p.fDetR;
	Vec3 u(p.fDetUX, p.fDetUY, p.fDetUZ); // u (tangential) direction
	Vec3 v(p.fDetVX, p.fDetVY, p.fDetVZ); // v (axial) direction
	Vec3 s(p.fSrcX, p.fSrcY, p.fSrcZ);    // source
	Vec3 d(p.fDetCX, p.fDetCY, p.fDetCZ); // center of detector

	//double fDetUT = u.norm() / R; // angular increment

	cyla = -cross3(u, v) * (R / (u.norm() * v.norm())); // radial direction

	if ((d - cyla - s).norm() > (d + cyla - s).norm())
		cyla = cyla * -1.0;

	cylc = d - cyla;                                    // center of cylinder
	cylb = u * (R / u.norm());                          // tangential direction
	//Vec3 cylaxis_n = v * (1.0 / v.norm());
	cylaxis = v;
}





// adjust pProjs to normalize volume geometry (translate + scale)
template<typename ProjectionT>
static bool convertAstraGeometry_internal(const CVolumeGeometry3D* pVolGeom,
                          std::vector<ProjectionT>& projs,
                          SVolScale3D& volScale)
{
	assert(pVolGeom);

	float dx = -(pVolGeom->getWindowMinX() + pVolGeom->getWindowMaxX()) / 2;
	float dy = -(pVolGeom->getWindowMinY() + pVolGeom->getWindowMaxY()) / 2;
	float dz = -(pVolGeom->getWindowMinZ() + pVolGeom->getWindowMaxZ()) / 2;

	float fx = 1.0f / pVolGeom->getPixelLengthX();
	float fy = 1.0f / pVolGeom->getPixelLengthY();
	float fz = 1.0f / pVolGeom->getPixelLengthZ();

	for (size_t i = 0; i < projs.size(); ++i) {
		projs[i].translate(dx, dy, dz);
		projs[i].scale(fx, fy, fz);
	}

	volScale.fX = pVolGeom->getPixelLengthX();
	volScale.fY = pVolGeom->getPixelLengthY();
	volScale.fZ = pVolGeom->getPixelLengthZ();

	return true;
}

// adjust pProjs to normalize volume geometry (translate only)
template<typename ProjectionT>
static bool convertAstraGeometry_unscaled_internal(const CVolumeGeometry3D* pVolGeom,
                          std::vector<ProjectionT>& projs,
                          SVolScale3D& volScale)
{
	assert(pVolGeom);

	float dx = -(pVolGeom->getWindowMinX() + pVolGeom->getWindowMaxX()) / 2;
	float dy = -(pVolGeom->getWindowMinY() + pVolGeom->getWindowMaxY()) / 2;
	float dz = -(pVolGeom->getWindowMinZ() + pVolGeom->getWindowMaxZ()) / 2;

	for (size_t i = 0; i < projs.size(); ++i) {
		projs[i].translate(dx, dy, dz);
	}

	// TODO: Check consistency of use of these values
    volScale.fX = pVolGeom->getPixelLengthX();
	volScale.fY = pVolGeom->getPixelLengthY();
	volScale.fZ = pVolGeom->getPixelLengthZ();

	return true;
}



bool convertAstraGeometry_dims(const CVolumeGeometry3D* pVolGeom,
                               const CProjectionGeometry3D* pProjGeom,
                               SDimensions3D& dims)
{
	dims.iVolX = pVolGeom->getGridColCount();
	dims.iVolY = pVolGeom->getGridRowCount();
	dims.iVolZ = pVolGeom->getGridSliceCount();
	dims.iProjAngles = pProjGeom->getProjectionCount();
	dims.iProjU = pProjGeom->getDetectorColCount();
	dims.iProjV = pProjGeom->getDetectorRowCount();

	if (dims.iVolX <= 0 || dims.iVolX <= 0 || dims.iVolX <= 0)
		return false;
	if (dims.iProjAngles <= 0 || dims.iProjU <= 0 || dims.iProjV <= 0)
		return false;

	return true;
}


static bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CParallelProjectionGeometry3D* pProjGeom,
                          std::vector<SPar3DProjection>& projs, SVolScale3D& volScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionAngles());

	int nth = pProjGeom->getProjectionCount();

	projs = genPar3DProjections(nth,
	                             pProjGeom->getDetectorColCount(),
	                             pProjGeom->getDetectorRowCount(),
	                             pProjGeom->getDetectorSpacingX(),
	                             pProjGeom->getDetectorSpacingY(),
	                             pProjGeom->getProjectionAngles());

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, projs, volScale);

	return ok;
}

static bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CParallelVecProjectionGeometry3D* pProjGeom,
                          std::vector<SPar3DProjection>& projs, SVolScale3D& volScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	int nth = pProjGeom->getProjectionCount();

	projs.resize(nth);
	for (int i = 0; i < nth; ++i)
		projs[i] = pProjGeom->getProjectionVectors()[i];

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, projs, volScale);

	return ok;
}

static bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CConeProjectionGeometry3D* pProjGeom,
                          std::vector<SConeProjection>& projs, SVolScale3D& volScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionAngles());

	int nth = pProjGeom->getProjectionCount();

	projs = genConeProjections(nth,
	                            pProjGeom->getDetectorColCount(),
	                            pProjGeom->getDetectorRowCount(),
	                            pProjGeom->getOriginSourceDistance(),
	                            pProjGeom->getOriginDetectorDistance(),
	                            pProjGeom->getDetectorSpacingX(),
	                            pProjGeom->getDetectorSpacingY(),
	                            pProjGeom->getProjectionAngles());

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, projs, volScale);

	return ok;
}

static bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CConeVecProjectionGeometry3D* pProjGeom,
                          std::vector<SConeProjection>& projs, SVolScale3D& volScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	int nth = pProjGeom->getProjectionCount();

	projs.resize(nth);
	for (int i = 0; i < nth; ++i)
		projs[i] = pProjGeom->getProjectionVectors()[i];

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, projs, volScale);

	return ok;
}

static bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CCylConeVecProjectionGeometry3D* pProjGeom,
                          std::vector<SCylConeProjection>& projs, SVolScale3D& volScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	int nth = pProjGeom->getProjectionCount();

	projs.resize(nth);
	for (int i = 0; i < nth; ++i)
		projs[i] = pProjGeom->getProjectionVectors()[i];

	bool ok;

	ok = convertAstraGeometry_unscaled_internal(pVolGeom, projs, volScale);

	return ok;
}

Geometry3DParameters convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                                        const CProjectionGeometry3D* pProjGeom,
                                        SVolScale3D& volScale)
{
	const CConeProjectionGeometry3D* conegeom = dynamic_cast<const CConeProjectionGeometry3D*>(pProjGeom);
	const CParallelProjectionGeometry3D* par3dgeom = dynamic_cast<const CParallelProjectionGeometry3D*>(pProjGeom);
	const CParallelVecProjectionGeometry3D* parvec3dgeom = dynamic_cast<const CParallelVecProjectionGeometry3D*>(pProjGeom);
	const CConeVecProjectionGeometry3D* conevec3dgeom = dynamic_cast<const CConeVecProjectionGeometry3D*>(pProjGeom);
	const CCylConeVecProjectionGeometry3D* cylconevec3dgeom = dynamic_cast<const CCylConeVecProjectionGeometry3D*>(pProjGeom);

	bool ok;


	if (conegeom || conevec3dgeom) {
		std::vector<SConeProjection> coneProjs;
		if (conegeom)
			ok = convertAstraGeometry(pVolGeom, conegeom, coneProjs, volScale);
		else
			ok = convertAstraGeometry(pVolGeom, conevec3dgeom, coneProjs, volScale);

		if (ok)
			return Geometry3DParameters::variant_t(std::move(coneProjs));
		else
			return Geometry3DParameters::variant_t();
	} else if (par3dgeom || parvec3dgeom) {
		std::vector<SPar3DProjection> parProjs;
		if (par3dgeom)
			ok = convertAstraGeometry(pVolGeom, par3dgeom, parProjs, volScale);
		else
			ok = convertAstraGeometry(pVolGeom, parvec3dgeom, parProjs, volScale);

		if (ok)
			return Geometry3DParameters::variant_t(std::move(parProjs));
		else
			return Geometry3DParameters::variant_t();
	} else if (cylconevec3dgeom) {
		std::vector<SCylConeProjection> cylConeProjs;
		ok = convertAstraGeometry(pVolGeom, cylconevec3dgeom, cylConeProjs, volScale);
		if (ok)
			return Geometry3DParameters::variant_t(std::move(cylConeProjs));
		else
			return Geometry3DParameters::variant_t();
	} else {
		ok = false;
	}

	return Geometry3DParameters::variant_t();
}





}
