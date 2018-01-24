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

#include "astra/GeometryUtil2D.h"

#include <cmath>

namespace astra {

SParProjection* genParProjections(unsigned int iProjAngles,
                                  unsigned int iProjDets,
                                  double fDetSize,
                                  const float *pfAngles,
                                  const float *pfExtraOffsets)
{
	SParProjection base;
	base.fRayX = 0.0f;
	base.fRayY = 1.0f;

	base.fDetSX = iProjDets * fDetSize * -0.5f;
	base.fDetSY = 0.0f;

	base.fDetUX = fDetSize;
	base.fDetUY = 0.0f;

	SParProjection* p = new SParProjection[iProjAngles];

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); } while(0)

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		if (pfExtraOffsets) {
			// TODO
		}

		ROTATE0(Ray, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);

		if (pfExtraOffsets) {
			float d = pfExtraOffsets[i];
			p[i].fDetSX -= d * p[i].fDetUX;
			p[i].fDetSY -= d * p[i].fDetUY;
		}
	}

#undef ROTATE0

	return p;
}


SFanProjection* genFanProjections(unsigned int iProjAngles,
                                  unsigned int iProjDets,
                                  double fOriginSource, double fOriginDetector,
                                  double fDetSize,
                                  const float *pfAngles)
//                                  const float *pfExtraOffsets)
{
	SFanProjection *pProjs = new SFanProjection[iProjAngles];

	float fSrcX0 = 0.0f;
	float fSrcY0 = -fOriginSource;
	float fDetUX0 = fDetSize;
	float fDetUY0 = 0.0f;
	float fDetSX0 = iProjDets * fDetUX0 / -2.0f;
	float fDetSY0 = fOriginDetector;

#define ROTATE0(name,i,alpha) do { pProjs[i].f##name##X = f##name##X0 * cos(alpha) - f##name##Y0 * sin(alpha); pProjs[i].f##name##Y = f##name##X0 * sin(alpha) + f##name##Y0 * cos(alpha); } while(0)
	for (unsigned int i = 0; i < iProjAngles; ++i) {
		ROTATE0(Src, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
	}

#undef ROTATE0

	return pProjs;
}

// Convert a SParProjection back into its set of "standard" circular parallel
// beam parameters. This is always possible.
bool getParParameters(const SParProjection &proj, unsigned int iProjDets, float &fAngle, float &fDetSize, float &fOffset)
{
	// Take part of DetU orthogonal to Ray
	double ux = proj.fDetUX;
	double uy = proj.fDetUY;

	double t = (ux * proj.fRayX + uy * proj.fRayY) / (proj.fRayX * proj.fRayX + proj.fRayY * proj.fRayY);

	ux -= t * proj.fRayX;
	uy -= t * proj.fRayY;

	double angle = atan2(uy, ux);

	fAngle = (float)angle;

	double norm2 = uy * uy + ux * ux;

	fDetSize = (float)sqrt(norm2);

	// CHECKME: SIGNS?
	fOffset = (float)(-0.5*iProjDets - (proj.fDetSY*uy + proj.fDetSX*ux) / norm2);

	return true;
}

// Convert a SFanProjection back into its set of "standard" circular fan beam
// parameters. This will return false if it can not be represented in this way.
bool getFanParameters(const SFanProjection &proj, unsigned int iProjDets, float &fAngle, float &fOriginSource, float &fOriginDetector, float &fDetSize, float &fOffset)
{
	// angle
	// det size
	// offset
	// origin-source
	// origin-detector

	// Need to check if line source-origin is orthogonal to vector ux,uy
	// (including the case source==origin)

	// (equivalent: source and origin project to same point on detector)

	double dp = proj.fSrcX * proj.fDetUX + proj.fSrcY * proj.fDetUY;

	double rel = (proj.fSrcX*proj.fSrcX + proj.fSrcY*proj.fSrcY) * (proj.fDetUX*proj.fDetUX + proj.fDetUY*proj.fDetUY);
	rel = sqrt(rel);

	if (std::abs(dp) > rel * 0.0001)
		return false;

	fOriginSource = sqrt(proj.fSrcX*proj.fSrcX + proj.fSrcY*proj.fSrcY);

	fDetSize = sqrt(proj.fDetUX*proj.fDetUX + proj.fDetUY*proj.fDetUY);

	// project origin on detector line ( == project source on detector line)

	double t = (- proj.fDetSX) * proj.fDetUX + (- proj.fDetSY) * proj.fDetUY;

	fOffset = (float)t - 0.5*iProjDets;

	// TODO: CHECKME
	fOriginDetector = sqrt((proj.fDetSX + t * proj.fDetUX)*(proj.fDetSX + t * proj.fDetUX) + (proj.fDetSY + t * proj.fDetUY)*(proj.fDetSY + t * proj.fDetUY));

	//float fAngle = atan2(proj.fDetSX + t * proj.fDetUX - proj.fSrcX, proj.fDetSY + t * proj.fDetUY); // TODO: Fix order + sign
	fAngle = atan2(proj.fDetUY, proj.fDetUX); // TODO: Check order + sign

	return true;
}


}
