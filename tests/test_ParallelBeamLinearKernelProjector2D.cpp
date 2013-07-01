/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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



#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "astra/ParallelBeamLineKernelProjector2D.h"
#include "astra/ParallelBeamLinearKernelProjector2D.h"
#include "astra/ParallelBeamStripKernelProjector2D.h"
#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/VolumeGeometry2D.h"
#include "astra/ProjectionGeometry2D.h"

#include <ctime>

using astra::float32;

struct TestParallelBeamLinearKernelProjector2D {
        TestParallelBeamLinearKernelProjector2D()
	{
		astra::float32 angles[] = { 2.6f };
		BOOST_REQUIRE( projGeom.initialize(1, 3, 1.0f, angles) );
		BOOST_REQUIRE( volGeom.initialize(3, 2) );
		BOOST_REQUIRE( proj.initialize(&projGeom, &volGeom) );
	}
        ~TestParallelBeamLinearKernelProjector2D()
	{

	}

	astra::CParallelBeamLinearKernelProjector2D proj;
	astra::CParallelProjectionGeometry2D projGeom;
	astra::CVolumeGeometry2D volGeom;
};

BOOST_FIXTURE_TEST_CASE( testParallelBeamLinearKernelProjector2D_General, TestParallelBeamLinearKernelProjector2D )
{

}


// Compute linear kernel for a single volume pixel/detector pixel combination
float32 compute_linear_kernel(const astra::CProjectionGeometry2D& projgeom, const astra::CVolumeGeometry2D& volgeom,
                  int iX, int iY, int iDet, float32 fAngle)
{
	// projection of center of volume pixel on detector array
	float32 fDetProj = (iX - (volgeom.getGridColCount()-1.0f)/2.0f ) * volgeom.getPixelLengthX() * cos(fAngle) - (iY - (volgeom.getGridRowCount()-1.0f)/2.0f ) * volgeom.getPixelLengthY() * sin(fAngle);

	// start of detector pixel on detector array
	float32 fDetStart = projgeom.indexToDetectorOffset(iDet) - 0.5f;

//	printf("(%d,%d,%d): %f in (%f,%f)\n", iX,iY,iDet,fDetProj, fDetStart, fDetStart+1.0f);

	// projection of center of next volume pixel on detector array
	float32 fDetStep;
	// length of projection ray through volume pixel
	float32 fWeight;

	if (fabs(cos(fAngle)) > fabs(sin(fAngle))) {
		fDetStep = volgeom.getPixelLengthY() * fabs(cos(fAngle));
		fWeight = volgeom.getPixelLengthX() * 1.0f / fabs(cos(fAngle));
	} else {
		fDetStep = volgeom.getPixelLengthX() * fabs(sin(fAngle));
		fWeight = volgeom.getPixelLengthY() * 1.0f / fabs(sin(fAngle));
	}

//	printf("step: %f\n   weight: %f\n", fDetStep, fWeight);

	// center of detector pixel on detector array
	float32 fDetCenter = fDetStart + 0.5f;

	// unweighted contribution of this volume pixel:
	// linear interpolation between
	//  fDetCenter - fDetStep    |---> 0
	//  fDetCenter               |---> 1
	//  fDetCenter + fDetStep    |---> 0
	float32 fBase;
	if (fDetCenter <= fDetProj) {
		fBase = (fDetCenter - (fDetProj - fDetStep))/fDetStep;
	} else {
		fBase = ((fDetProj + fDetStep) - fDetCenter)/fDetStep;
	}
//	printf("base: %f\n", fBase);
	if (fBase < 0) fBase = 0;
	return fBase * fWeight;
}

BOOST_AUTO_TEST_CASE( testParallelBeamLinearKernelProjector2D_Rectangles )
{
	astra::CParallelBeamLinearKernelProjector2D proj;
	astra::CParallelProjectionGeometry2D projGeom;
	astra::CVolumeGeometry2D volGeom;

	const unsigned int iRandomTestCount = 100;

	unsigned int iSeed = time(0);
	srand(iSeed);

	for (unsigned int iTest = 0; iTest < iRandomTestCount; ++iTest) {
		int iDetectorCount = 1 + (rand() % 100);
		int iRows = 1 + (rand() % 100);
		int iCols = 1 + (rand() % 100);
		
		
		astra::float32 angles[] = { rand() * 2.0f*astra::PI / RAND_MAX };
		projGeom.initialize(1, iDetectorCount, 0.8f, angles);
		volGeom.initialize(iCols, iRows);
		proj.initialize(&projGeom, &volGeom);

		int iMax = proj.getProjectionWeightsCount(0);
		BOOST_REQUIRE(iMax > 0);

		astra::SPixelWeight* pPix = new astra::SPixelWeight[iMax];
		BOOST_REQUIRE(pPix);

		astra::float32 fWeight = 0;
		for (int iDet = 0; iDet < projGeom.getDetectorCount(); ++iDet) {
			int iCount;
			proj.computeSingleRayWeights(0, iDet, pPix, iMax, iCount); 
			BOOST_REQUIRE(iCount <= iMax);

			astra::float32 fW = 0;
			for (int i = 0; i < iCount; ++i) {
				float32 fTest = compute_linear_kernel(
				            projGeom,
				            volGeom,
				            pPix[i].m_iIndex % volGeom.getGridColCount(),
				            pPix[i].m_iIndex / volGeom.getGridColCount(),
				            iDet,
				            projGeom.getProjectionAngle(0));
				BOOST_CHECK_SMALL( pPix[i].m_fWeight - fTest, 0.00037f);
				fW += pPix[i].m_fWeight;
			}

			fWeight += fW;

		}

		delete[] pPix;
	}
}


