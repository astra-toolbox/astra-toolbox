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

#include "astra/Float32VolumeData2D.h"

// Note: most of the features of CFloat32VolumeData2D are tested by
// the CFloat32Data2D tests.

BOOST_AUTO_TEST_CASE( testFloat32VolumeData2D_Constructor1 )
{
	astra::CVolumeGeometry2D geom(16, 32);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32VolumeData2D data(&geom);
	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getType() == astra::CFloat32Data2D::VOLUME );
	BOOST_CHECK( data.getGeometry()->isEqual(&geom) );
}

BOOST_AUTO_TEST_CASE( testFloat32VolumeData2D_Constructor1odd )
{
	astra::CVolumeGeometry2D geom(16, 32);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32VolumeData2D data(&geom, 1.0f);
	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getType() == astra::CFloat32Data2D::VOLUME );
	BOOST_CHECK( data.getGeometry()->isEqual(&geom) );

	// CHECKME: should this be necessary?
	data.updateStatistics();
	BOOST_CHECK( data.getGlobalMax() == 1.0f );
}

BOOST_AUTO_TEST_CASE( testFloat32VolumeData2D_Constructor2 )
{
	astra::float32 d[] = { 1.0f, 2.0f, 3.0f, 4.0f };
	astra::CVolumeGeometry2D geom(2, 2);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32VolumeData2D data(&geom, d);
	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getType() == astra::CFloat32Data2D::VOLUME );

	BOOST_CHECK( data.getGeometry()->isEqual(&geom) );

	// CHECKME: should this be necessary?
	data.updateStatistics();
	BOOST_CHECK( data.getGlobalMax() == 4.0f );
}

BOOST_AUTO_TEST_CASE( testFloat32VolumeData2D_Clone )
{
	astra::float32 d[] = { 1.0f, 2.0f, 3.0f, 4.0f };
	astra::CVolumeGeometry2D geom(2, 2);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32VolumeData2D data(&geom, d);
	BOOST_REQUIRE( data.isInitialized() );

	astra::CFloat32VolumeData2D data2(data);
	BOOST_REQUIRE( data2.isInitialized() );

	BOOST_CHECK( data2.getGeometry()->isEqual(&geom) );
	BOOST_CHECK( data2.getDataConst()[0] == 1.0f );
	BOOST_CHECK( data2.getDataConst()[3] == 4.0f );

	astra::CFloat32VolumeData2D data3;
	data3 = data;
	BOOST_REQUIRE( data3.isInitialized() );

	BOOST_CHECK( data3.getGeometry()->isEqual(&geom) );
	BOOST_CHECK( data3.getDataConst()[0] == 1.0f );
	BOOST_CHECK( data3.getDataConst()[3] == 4.0f );
}
