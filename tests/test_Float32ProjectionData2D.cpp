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

#include "astra/Float32ProjectionData2D.h"
#include "astra/ParallelProjectionGeometry2D.h"

// Note: most of the features of CFloat32ProjectionData2D are tested by
// the CFloat32Data2D tests.

BOOST_AUTO_TEST_CASE( testFloat32ProjectionData2D_Constructor1 )
{
	astra::float32 angles[] = { 0.0f, 1.0f };
	astra::CParallelProjectionGeometry2D geom(2, 4, 0.5f, angles);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32ProjectionData2D data(&geom);
	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getType() == astra::CFloat32Data2D::PROJECTION );
	BOOST_CHECK( data.getGeometry()->isEqual(&geom) );
}

BOOST_AUTO_TEST_CASE( testFloat32ProjectionData2D_Constructor2 )
{
	astra::float32 angles[] = { 0.0f, 1.0f };
	astra::float32 d[] = { 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.f };
	astra::CParallelProjectionGeometry2D geom(2, 4, 0.5f, angles);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32ProjectionData2D data(&geom, d);
	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getType() == astra::CFloat32Data2D::PROJECTION );
	BOOST_CHECK( data.getGeometry()->isEqual(&geom) );

	// CHECKME: should this be necessary?
	data.updateStatistics();

	BOOST_CHECK( data.getGlobalMax() == 10.0f );
}

BOOST_AUTO_TEST_CASE( testFloat32ProjectionData2D_Constructor3 )
{
	astra::float32 angles[] = { 0.0f, 1.0f };
	astra::CParallelProjectionGeometry2D geom(2, 4, 0.5f, angles);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32ProjectionData2D data(&geom, 3.5f);
	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getType() == astra::CFloat32Data2D::PROJECTION );
	BOOST_CHECK( data.getGeometry()->isEqual(&geom) );

	// CHECKME: should this be necessary?
	data.updateStatistics();

	BOOST_CHECK( data.getGlobalMax() == 3.5f );
}

BOOST_AUTO_TEST_CASE( testFloat32ProjectionData2D_Clone )
{
	astra::float32 angles[] = { 0.0f, 1.0f };
	astra::CParallelProjectionGeometry2D geom(2, 4, 0.5f, angles);
	BOOST_REQUIRE( geom.isInitialized() );

	astra::CFloat32ProjectionData2D data(&geom, 3.5f);
	BOOST_REQUIRE( data.isInitialized() );

	astra::CFloat32ProjectionData2D data2(data);
	BOOST_REQUIRE( data2.isInitialized() );

	BOOST_CHECK( data2.getGeometry()->isEqual(&geom) );
	BOOST_CHECK( data2.getDataConst()[0] == 3.5f );
	BOOST_CHECK( data2.getDataConst()[3] == 3.5f );

	astra::CFloat32ProjectionData2D data3;
	data3 = data;
	BOOST_REQUIRE( data3.isInitialized() );

	BOOST_CHECK( data3.getGeometry()->isEqual(&geom) );
	BOOST_CHECK( data3.getDataConst()[0] == 3.5f );
	BOOST_CHECK( data3.getDataConst()[3] == 3.5f );
}
