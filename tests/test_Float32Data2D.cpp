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

#include "astra/Float32Data2D.h"


// Utility class to test Float32Data2D
class CTestFloat32Data2D : public astra::CFloat32Data2D {
public:
	CTestFloat32Data2D(int _iWidth, int _iHeight)
	{
		m_bInitialized = _initialize(_iWidth, _iHeight);
	}
	CTestFloat32Data2D(int _iWidth, int _iHeight, const astra::float32* _pfData)
	{
		m_bInitialized = _initialize(_iWidth, _iHeight, _pfData);
	}

	CTestFloat32Data2D(int _iWidth, int _iHeight, astra::float32 _fScalar)
	{
		m_bInitialized = _initialize(_iWidth, _iHeight, _fScalar);
	}

	CTestFloat32Data2D() { }

};


struct TestFloat32Data2D {
	static astra::float32 d[];
	TestFloat32Data2D() : data(2,2,d)
	{
	}

	~TestFloat32Data2D()
	{
	}

	CTestFloat32Data2D data;
};

astra::float32 TestFloat32Data2D::d[] = { 1.0f, 2.0f, 3.0f, 4.0f };

BOOST_AUTO_TEST_CASE( testFloat32Data2D_Constructor1 )
{
	CTestFloat32Data2D data(2,2);

	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getWidth() == 2 );
	BOOST_CHECK( data.getHeight() == 2 );
	BOOST_CHECK( data.getSize() == 4 );
	BOOST_CHECK( data.getDimensionCount() == 2 );

	BOOST_REQUIRE( data.getDataConst() != 0 );
	BOOST_REQUIRE( data.getData() != 0 );
	BOOST_REQUIRE( data.getData2D() != 0 );
	BOOST_REQUIRE( data.getData2DConst() != 0 );
	BOOST_REQUIRE( data.getData2D()[0] != 0 );
	BOOST_REQUIRE( data.getData2DConst()[0] != 0 );

	data.setData(1.0f);

	// CHECKME: should this be necessary?
	data.updateStatistics();

	BOOST_CHECK( data.getGlobalMin() == 1.0f );
	BOOST_CHECK( data.getGlobalMax() == 1.0f );
	BOOST_CHECK( data.getGlobalMean() == 1.0f );

	BOOST_CHECK( data.getData()[0] == 1.0f );
	BOOST_CHECK( data.getDataConst()[0] == 1.0f );
	BOOST_CHECK( data.getData2D()[0][0] == 1.0f );
	BOOST_CHECK( data.getData2DConst()[0][0] == 1.0f );
}

BOOST_AUTO_TEST_CASE( testFloat32Data2D_Constructor2 )
{
	CTestFloat32Data2D data(2,2,1.5f);

	// CHECKME: should this be necessary?
	data.updateStatistics();

	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getWidth() == 2 );
	BOOST_CHECK( data.getHeight() == 2 );
	BOOST_CHECK( data.getSize() == 4 );
	BOOST_CHECK( data.getDimensionCount() == 2 );

	BOOST_REQUIRE( data.getDataConst() != 0 );
	BOOST_REQUIRE( data.getData() != 0 );
	BOOST_REQUIRE( data.getData2D() != 0 );
	BOOST_REQUIRE( data.getData2DConst() != 0 );
	BOOST_REQUIRE( data.getData2D()[0] != 0 );
	BOOST_REQUIRE( data.getData2DConst()[0] != 0 );

	BOOST_CHECK( data.getGlobalMin() == 1.5f );
	BOOST_CHECK( data.getGlobalMax() == 1.5f );
	BOOST_CHECK( data.getGlobalMean() == 1.5f );

	BOOST_CHECK( data.getData()[0] == 1.5f );
	BOOST_CHECK( data.getDataConst()[0] == 1.5f );
	BOOST_CHECK( data.getData2D()[0][0] == 1.5f );
	BOOST_CHECK( data.getData2DConst()[0][0] == 1.5f );
}

BOOST_AUTO_TEST_CASE( testFloat32Data2D_Constructor3 )
{
	astra::float32 d[] = { 1.0f, 2.0f, 3.0f, 4.0f };
	CTestFloat32Data2D data(2,2,d);

	// CHECKME: should this be necessary?
	data.updateStatistics();

	BOOST_REQUIRE( data.isInitialized() );

	BOOST_CHECK( data.getWidth() == 2 );
	BOOST_CHECK( data.getHeight() == 2 );
	BOOST_CHECK( data.getSize() == 4 );
	BOOST_CHECK( data.getDimensionCount() == 2 );

	BOOST_REQUIRE( data.getDataConst() != 0 );
	BOOST_REQUIRE( data.getData() != 0 );
	BOOST_REQUIRE( data.getData2D() != 0 );
	BOOST_REQUIRE( data.getData2DConst() != 0 );
	BOOST_REQUIRE( data.getData2D()[0] != 0 );
	BOOST_REQUIRE( data.getData2DConst()[0] != 0 );

	BOOST_CHECK( data.getGlobalMin() == 1.0f );
	BOOST_CHECK( data.getGlobalMax() == 4.0f );

	BOOST_CHECK( data.getData()[0] == 1.0f );
	BOOST_CHECK( data.getDataConst()[0] == 1.0f );
	BOOST_CHECK( data.getData2D()[0][0] == 1.0f );
	BOOST_CHECK( data.getData2DConst()[0][0] == 1.0f );

}

BOOST_FIXTURE_TEST_CASE( testFloat32Data2D_Operators, TestFloat32Data2D )
{
	// Note: all operations below involve exactly representable floats,
	// so there is no need to use epsilons in the checks

	data.updateStatistics();

	// FIXME: should those be correct here?
	BOOST_CHECK( data.getGlobalMin() == 1.0f );
	BOOST_CHECK( data.getGlobalMax() == 4.0f );
	BOOST_CHECK( data.getGlobalMean() == 2.5f );

	data *= 2.0f;

	BOOST_CHECK( data.getDataConst()[0] == 2.0f );
	BOOST_CHECK( data.getDataConst()[3] == 8.0f );

	data /= 0.5f;

	BOOST_CHECK( data.getDataConst()[0] == 4.0f );
	BOOST_CHECK( data.getDataConst()[3] == 16.0f );

	astra::float32 d[] = { 1.0f, 2.0f, 3.0f, 4.0f };
	CTestFloat32Data2D data2(2,2,d);

	data += data2;

	BOOST_CHECK( data.getDataConst()[0] == 5.0f );
	BOOST_CHECK( data.getDataConst()[3] == 20.0f );
	
	data *= data2;
	
	BOOST_CHECK( data.getDataConst()[0] == 5.0f );
	BOOST_CHECK( data.getDataConst()[3] == 80.0f );

	data -= data2;
	BOOST_CHECK( data.getDataConst()[0] == 4.0f );
	BOOST_CHECK( data.getDataConst()[3] == 76.0f );

	data += 0.5f;
	BOOST_CHECK( data.getDataConst()[0] == 4.5f );
	BOOST_CHECK( data.getDataConst()[3] == 76.5f );

	data -= 0.5f;
	BOOST_CHECK( data.getDataConst()[0] == 4.0f );
	BOOST_CHECK( data.getDataConst()[3] == 76.0f );
}

BOOST_FIXTURE_TEST_CASE( testFloat32Data2D_Update, TestFloat32Data2D )
{
	data.getData()[2] = 42.0f;
	data.getData()[1] = -37.0f;
	data.updateStatistics();

	BOOST_CHECK( data.getGlobalMin() == -37.0f );
	BOOST_CHECK( data.getGlobalMax() == 42.0f );
	BOOST_CHECK( data.getGlobalMean() == 2.5f );
}
