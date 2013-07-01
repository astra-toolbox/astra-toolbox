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
#include <boost/test/test_case_template.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cmath>

#include "astra/FanFlatProjectionGeometry2D.h"

BOOST_AUTO_TEST_CASE( testFanFlatProjectionGeometry2D_Constructor )
{
	astra::float32 angles[] = { 0.0f, 1.0f, 2.0f, 3.0f };
	astra::CFanFlatProjectionGeometry2D geom(4, 8, 0.5f, angles, 1.0f, 2.0f);

	BOOST_REQUIRE( geom.isInitialized() );

	BOOST_CHECK( geom.getProjectionAngleCount() == 4 );
	BOOST_CHECK( geom.getDetectorCount() == 8 );
	BOOST_CHECK( geom.getDetectorWidth() == 0.5f );
	BOOST_CHECK( geom.getProjectionAngle(0) == 0.0f );
	BOOST_CHECK( geom.getProjectionAngle(1) == 1.0f );
	BOOST_CHECK( geom.getProjectionAngle(2) == 2.0f );
	BOOST_CHECK( geom.getProjectionAngle(3) == 3.0f );
	BOOST_CHECK( geom.getProjectionAngles()[0] == 0.0f );
	BOOST_CHECK( geom.getProjectionAngles()[3] == 3.0f );
	BOOST_CHECK( geom.getOriginSourceDistance() == 1.0f );
	BOOST_CHECK( geom.getOriginDetectorDistance() == 2.0f );
}

BOOST_AUTO_TEST_CASE( testFanFlatProjectionGeometry2D_Offsets )
{
	astra::float32 angles[] = { 0.0f, 1.0f, 2.0f, 3.0f };
	astra::CFanFlatProjectionGeometry2D geom(4, 8, 0.5f, angles, 1.0f, 2.0f);

	BOOST_REQUIRE( geom.isInitialized() );

	BOOST_CHECK( geom.getSourceDetectorDistance() == 3.0f );
	BOOST_CHECK_SMALL( geom.getProjectionAngleDegrees(2) - 114.591559026165f, 1e-5f );

	// CHECKME: where is the center of the detector array?
	BOOST_CHECK( geom.detectorOffsetToIndexFloat(0.0f) == 3.5f );
	BOOST_CHECK( geom.detectorOffsetToIndexFloat(0.625f) == 4.75f );
	BOOST_CHECK( geom.detectorOffsetToIndex(-0.1f) == 3 );

	BOOST_CHECK( geom.indexToDetectorOffset(0) == -1.75f );
	BOOST_CHECK( geom.indexToDetectorOffset(1) == -1.25f );

	int angle, detector;
	geom.indexToAngleDetectorIndex(10, angle, detector);
	BOOST_CHECK( angle == 1 );
	BOOST_CHECK( detector == 2 );

	float t, theta;
	geom.getRayParams(0, 2, t, theta);
	BOOST_CHECK_SMALL( tan(theta) + 0.25f, astra::eps );
	BOOST_CHECK_SMALL( 17.0f*t*t - 1.0f, astra::eps );

	// TODO: add test with large angle
}


BOOST_AUTO_TEST_CASE( testFanFlatProjectionGeometry2D_Clone )
{
	astra::float32 angles[] = { 0.0f, 1.0f, 2.0f, 3.0f };
	astra::CFanFlatProjectionGeometry2D geom(4, 8, 0.5f, angles, 1.0f, 2.0f);

	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFanFlatProjectionGeometry2D* geom2;
	geom2 = dynamic_cast<astra::CFanFlatProjectionGeometry2D*>(geom.clone());

	BOOST_REQUIRE( geom2 );
	BOOST_REQUIRE( geom2->isInitialized() );

	BOOST_CHECK( geom.isEqual(geom2) );
	BOOST_CHECK( geom2->getProjectionAngleCount() == 4 );
	BOOST_CHECK( geom2->getDetectorCount() == 8 );
	BOOST_CHECK( geom2->getDetectorWidth() == 0.5f );
	BOOST_CHECK( geom2->getProjectionAngle(0) == 0.0f );
	BOOST_CHECK( geom2->getProjectionAngle(1) == 1.0f );
	BOOST_CHECK( geom2->getProjectionAngle(2) == 2.0f );
	BOOST_CHECK( geom2->getProjectionAngle(3) == 3.0f );
	BOOST_CHECK( geom2->getProjectionAngles()[0] == 0.0f );
	BOOST_CHECK( geom2->getProjectionAngles()[3] == 3.0f );
	BOOST_CHECK( geom2->getOriginSourceDistance() == 1.0f );
	BOOST_CHECK( geom2->getOriginDetectorDistance() == 2.0f );
	delete geom2;
}

