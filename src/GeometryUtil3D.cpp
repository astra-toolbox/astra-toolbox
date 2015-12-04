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

#include "astra/GeometryUtil3D.h"

#include <cmath>

namespace astra {


SConeProjection* genConeProjections(unsigned int iProjAngles,
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

	SConeProjection* p = new SConeProjection[iProjAngles];

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

SPar3DProjection* genPar3DProjections(unsigned int iProjAngles,
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

	SPar3DProjection* p = new SPar3DProjection[iProjAngles];

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


// TODO: Handle cases of rays parallel to coordinate planes

void backprojectPointX(const SPar3DProjection& proj, double fU, double fV,
                       double fX, double &fY, double &fZ)
{
	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fX - px) / proj.fRayX;

	fY = py + a * proj.fRayY;
	fZ = pz + a * proj.fRayZ;
}

void backprojectPointY(const SPar3DProjection& proj, double fU, double fV,
                       double fY, double &fX, double &fZ)
{
	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fY - py) / proj.fRayY;

	fX = px + a * proj.fRayX;
	fZ = pz + a * proj.fRayZ;

}

void backprojectPointZ(const SPar3DProjection& proj, double fU, double fV,
                       double fZ, double &fX, double &fY)
{
	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fZ - pz) / proj.fRayZ;

	fX = px + a * proj.fRayX;
	fY = py + a * proj.fRayY;
}



void backprojectPointX(const SConeProjection& proj, double fU, double fV,
                       double fX, double &fY, double &fZ)
{
	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fX - proj.fSrcX) / (px - proj.fSrcX);

	fY = proj.fSrcY + a * (py - proj.fSrcY);
	fZ = proj.fSrcZ + a * (pz - proj.fSrcZ);
}

void backprojectPointY(const SConeProjection& proj, double fU, double fV,
                       double fY, double &fX, double &fZ)
{
	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fY - proj.fSrcY) / (py - proj.fSrcY);

	fX = proj.fSrcX + a * (px - proj.fSrcX);
	fZ = proj.fSrcZ + a * (pz - proj.fSrcZ);
}

void backprojectPointZ(const SConeProjection& proj, double fU, double fV,
                       double fZ, double &fX, double &fY)
{
	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fZ - proj.fSrcZ) / (pz - proj.fSrcZ);

	fX = proj.fSrcX + a * (px - proj.fSrcX);
	fY = proj.fSrcY + a * (py - proj.fSrcY);
}


}
