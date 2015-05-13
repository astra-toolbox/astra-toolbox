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
#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/VolumeGeometry2D.h"

struct TestParallelBeamLineKernelProjector2D {
        TestParallelBeamLineKernelProjector2D()
	{
		astra::float32 angles[] = { 1.0f };
		BOOST_REQUIRE( projGeom.initialize(1, 9, 1.0f, angles) );
		BOOST_REQUIRE( volGeom.initialize(6, 4) );
		BOOST_REQUIRE( proj.initialize(&projGeom, &volGeom) );
	}
        ~TestParallelBeamLineKernelProjector2D()
	{

	}

	astra::CParallelBeamLineKernelProjector2D proj;
	astra::CParallelProjectionGeometry2D projGeom;
	astra::CVolumeGeometry2D volGeom;
};

BOOST_FIXTURE_TEST_CASE( testParallelBeamLineKernelProjector2D_General, TestParallelBeamLineKernelProjector2D )
{

}

BOOST_FIXTURE_TEST_CASE( testParallelBeamLineKernelProjector2D_Rectangle, TestParallelBeamLineKernelProjector2D )
{
	int iMax = proj.getProjectionWeightsCount(0);
	BOOST_REQUIRE(iMax > 0);

	astra::SPixelWeight* pPix = new astra::SPixelWeight[iMax];
	BOOST_REQUIRE(pPix);

	int iCount;
	proj.computeSingleRayWeights(0, 4, pPix, iMax, iCount); 
	BOOST_REQUIRE(iCount <= iMax);

	astra::float32 fWeight = 0;
	for (int i = 0; i < iCount; ++i)
		fWeight += pPix[i].m_fWeight;

	BOOST_CHECK_SMALL(fWeight - 7.13037f, 0.00001f);

	delete[] pPix;
}


