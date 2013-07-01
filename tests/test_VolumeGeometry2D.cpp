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

#include "astra/VolumeGeometry2D.h"

BOOST_AUTO_TEST_CASE( testVolumeGeometry2D_Constructor1 )
{
	astra::CVolumeGeometry2D geom(16, 32);

	BOOST_REQUIRE( geom.isInitialized() );

	BOOST_CHECK( geom.getGridColCount() == 16 );
	BOOST_CHECK( geom.getGridRowCount() == 32 );
	BOOST_CHECK( geom.getGridTotCount() == 512 );
	BOOST_CHECK( geom.getWindowLengthX() == 16.0f );
	BOOST_CHECK( geom.getWindowLengthY() == 32.0f );
	BOOST_CHECK( geom.getWindowArea() == 512.0f );
	BOOST_CHECK( geom.getPixelLengthX() == 1.0f );
	BOOST_CHECK( geom.getPixelLengthY() == 1.0f );
	BOOST_CHECK( geom.getPixelArea() == 1.0f );
	BOOST_CHECK( geom.getWindowMinX() == -8.0f );
	BOOST_CHECK( geom.getWindowMaxX() == 8.0f );
	BOOST_CHECK( geom.getWindowMinY() == -16.0f );
	BOOST_CHECK( geom.getWindowMaxY() == 16.0f );
}

BOOST_AUTO_TEST_CASE( testVolumeGeometry2D_Constructor1odd )
{
	astra::CVolumeGeometry2D geom(5, 7);

	BOOST_REQUIRE( geom.isInitialized() );

	BOOST_CHECK( geom.getGridColCount() == 5 );
	BOOST_CHECK( geom.getGridRowCount() == 7 );
	BOOST_CHECK( geom.getGridTotCount() == 35 );
	BOOST_CHECK( geom.getWindowLengthX() == 5.0f );
	BOOST_CHECK( geom.getWindowLengthY() == 7.0f );
	BOOST_CHECK( geom.getWindowArea() == 35.0f );
	BOOST_CHECK( geom.getPixelLengthX() == 1.0f );
	BOOST_CHECK( geom.getPixelLengthY() == 1.0f );
	BOOST_CHECK( geom.getPixelArea() == 1.0f );
	BOOST_CHECK( geom.getWindowMinX() == -2.5f );
	BOOST_CHECK( geom.getWindowMaxX() == 2.5f );
	BOOST_CHECK( geom.getWindowMinY() == -3.5f );
	BOOST_CHECK( geom.getWindowMaxY() == 3.5f );
}

BOOST_AUTO_TEST_CASE( testVolumeGeometry2D_Constructor2 )
{
	astra::CVolumeGeometry2D geom(16, 32, -1.0f, -2.0f, 3.0f, 4.0f);

	BOOST_REQUIRE( geom.isInitialized() );

	BOOST_CHECK( geom.getGridColCount() == 16 );
	BOOST_CHECK( geom.getGridRowCount() == 32 );
	BOOST_CHECK( geom.getGridTotCount() == 512 );
	BOOST_CHECK( geom.getWindowLengthX() == 4.0f );
	BOOST_CHECK( geom.getWindowLengthY() == 6.0f );
	BOOST_CHECK( geom.getWindowArea() == 24.0f );
	BOOST_CHECK( geom.getPixelLengthX() == 0.25f );
	BOOST_CHECK( geom.getPixelLengthY() == 0.1875f );
	BOOST_CHECK( geom.getPixelArea() == 0.046875f );
	BOOST_CHECK( geom.getWindowMinX() == -1.0f );
	BOOST_CHECK( geom.getWindowMaxX() == 3.0f );
	BOOST_CHECK( geom.getWindowMinY() == -2.0f );
	BOOST_CHECK( geom.getWindowMaxY() == 4.0f );
}

BOOST_AUTO_TEST_CASE( testVolumeGeometry2D_Clone )
{
	astra::CVolumeGeometry2D geom(16, 32, -1.0f, -2.0f, 3.0f, 4.0f);

	astra::CVolumeGeometry2D* geom2 = geom.clone();

	BOOST_REQUIRE( geom2->isInitialized() );

	BOOST_CHECK( geom.isEqual(geom2) );
	BOOST_CHECK( geom2->getGridColCount() == 16 );
	BOOST_CHECK( geom2->getGridRowCount() == 32 );
	BOOST_CHECK( geom2->getGridTotCount() == 512 );
	BOOST_CHECK( geom2->getWindowLengthX() == 4.0f );
	BOOST_CHECK( geom2->getWindowLengthY() == 6.0f );
	BOOST_CHECK( geom2->getWindowArea() == 24.0f );
	BOOST_CHECK( geom2->getPixelLengthX() == 0.25f );
	BOOST_CHECK( geom2->getPixelLengthY() == 0.1875f );
	BOOST_CHECK( geom2->getPixelArea() == 0.046875f );
	BOOST_CHECK( geom2->getWindowMinX() == -1.0f );
	BOOST_CHECK( geom2->getWindowMaxX() == 3.0f );
	BOOST_CHECK( geom2->getWindowMinY() == -2.0f );
	BOOST_CHECK( geom2->getWindowMaxY() == 4.0f );

	delete geom2;
}

BOOST_AUTO_TEST_CASE( testVolumeGeometry2D_Offsets )
{
	astra::CVolumeGeometry2D geom(16, 32, -1.0f, -2.0f, 3.0f, 4.0f);

	BOOST_REQUIRE( geom.isInitialized() );

	BOOST_CHECK( geom.pixelRowColToIndex(1,2) == 18 );

	int r,c;
	geom.pixelIndexToRowCol(66, r, c);
	BOOST_CHECK( r == 4 );
	BOOST_CHECK( c == 2 );

	BOOST_CHECK( geom.pixelColToCenterX(2) == -0.375f );
	BOOST_CHECK( geom.pixelColToMinX(2) == -0.5f );
	BOOST_CHECK( geom.pixelColToMaxX(2) == -0.25f );

	BOOST_CHECK( geom.pixelRowToCenterY(29) == -1.53125f );
	BOOST_CHECK( geom.pixelRowToMinY(29) == -1.625f );
	BOOST_CHECK( geom.pixelRowToMaxY(29) == -1.4375f );

	BOOST_CHECK( geom.coordXToCol(0.1f) == 4 );
	BOOST_CHECK( geom.coordYToRow(0.1f) == 20 );
}
