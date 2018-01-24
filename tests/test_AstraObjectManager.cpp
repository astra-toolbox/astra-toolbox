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

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>

#include "astra/AstraObjectManager.h"

struct TestT {
	TestT(int _x) : x(_x) { }
	bool operator==(int _x) const { return x == _x; }

	int x;
	bool isInitialized() const { return true; }
	std::string description() const { return ""; }
};

namespace astra {

class CTestManager : public Singleton<CTestManager>, public CAstraObjectManager<TestT>
{
        virtual std::string getType() const { return "test"; }
};

DEFINE_SINGLETON(CTestManager);

}

BOOST_AUTO_TEST_CASE( testAstraObjectManager )
{
	astra::CTestManager &man = astra::CTestManager::getSingleton();

	int i1 = man.store(new TestT(1));
	BOOST_REQUIRE(man.hasIndex(i1));
	BOOST_CHECK(*(man.get(i1)) == 1);

	int i2 = man.store(new TestT(2));
	BOOST_REQUIRE(man.hasIndex(i2));
	BOOST_CHECK(*(man.get(i1)) == 1);
	BOOST_CHECK(*(man.get(i2)) == 2);

	man.remove(i1);

	BOOST_CHECK(!man.hasIndex(i1));
	BOOST_REQUIRE(man.hasIndex(i2));

	int i3 = man.store(new TestT(3));
	BOOST_REQUIRE(man.hasIndex(i3));
	BOOST_CHECK(*(man.get(i2)) == 2);
	BOOST_CHECK(*(man.get(i3)) == 3);

	TestT* pi4 = new TestT(4);
	int i4 = man.store(pi4);
	BOOST_REQUIRE(man.hasIndex(i4));
	BOOST_CHECK(*(man.get(i2)) == 2);
	BOOST_CHECK(*(man.get(i3)) == 3);
	BOOST_CHECK(*(man.get(i4)) == 4);

	BOOST_CHECK(man.getIndex(pi4) == i4);

	man.clear();

	BOOST_CHECK(!man.hasIndex(i1));
	BOOST_CHECK(!man.hasIndex(i2));
	BOOST_CHECK(!man.hasIndex(i3));
	BOOST_CHECK(!man.hasIndex(i4));
	BOOST_CHECK(!man.getIndex(pi4));
}
