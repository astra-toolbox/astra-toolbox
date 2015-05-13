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

#include "astra/XMLDocument.h"
#include "astra/Config.h"

BOOST_AUTO_TEST_CASE( testXMLDocument_Constructor1 )
{
	astra::XMLDocument *doc = astra::XMLDocument::createDocument("test");
	BOOST_REQUIRE(doc);

	astra::XMLNode root = doc->getRootNode();

	BOOST_REQUIRE(root);

	BOOST_CHECK(root.getName() == "test");
	BOOST_CHECK(root.getContent().empty());

	delete doc;

}

BOOST_AUTO_TEST_CASE( testXMLDocument_FileIO )
{
	astra::XMLDocument *doc = astra::XMLDocument::createDocument("test");

	doc->saveToFile("test.xml");

	astra::XMLDocument *doc2 = astra::XMLDocument::readFromFile("test.xml");
	astra::XMLNode root = doc2->getRootNode();

	BOOST_REQUIRE(root);

	BOOST_CHECK(root.getName() == "test");
	BOOST_CHECK(root.getContent().empty());

	delete doc2;
	delete doc;

}

BOOST_AUTO_TEST_CASE( testXMLDocument_CreateNodes )
{
	astra::XMLDocument *doc = astra::XMLDocument::createDocument("test");
	BOOST_REQUIRE(doc);

	astra::XMLNode root = doc->getRootNode();
	BOOST_REQUIRE(root);

	astra::XMLNode node = root.addChildNode("child");
	BOOST_REQUIRE(node);

	node.addAttribute("attr", "val");

	doc->saveToFile("test2.xml");

	delete doc;

	doc = astra::XMLDocument::readFromFile("test2.xml");
	BOOST_REQUIRE(doc);
	root = doc->getRootNode();
	BOOST_REQUIRE(node);
	node = root.getSingleNode("child");
	BOOST_REQUIRE(node);

	BOOST_CHECK(node.hasAttribute("attr"));
	BOOST_CHECK(node.getAttribute("attr") == "val");

	delete doc;
}

BOOST_AUTO_TEST_CASE( testXMLDocument_Options )
{
	astra::XMLDocument *doc = astra::XMLDocument::createDocument("test");
	BOOST_REQUIRE(doc);

	astra::XMLNode root = doc->getRootNode();
	BOOST_REQUIRE(root);

	BOOST_CHECK(!root.hasOption("opt"));

	root.addOption("opt", "val");

	BOOST_CHECK(root.hasOption("opt"));

	BOOST_CHECK(root.getOption("opt") == "val");

	delete doc;

}

BOOST_AUTO_TEST_CASE( testXMLDocument_List )
{
	astra::XMLDocument *doc = astra::XMLDocument::createDocument("test");
	BOOST_REQUIRE(doc);

	astra::XMLNode root = doc->getRootNode();
	BOOST_REQUIRE(root);

	astra::XMLNode node = root.addChildNode("child");
	BOOST_REQUIRE(node);


	float fl[] = { 1.0, 3.5, 2.0, 4.75 };

	node.setContent(fl, sizeof(fl)/sizeof(fl[0]));

	doc->saveToFile("test3.xml");

	delete doc;

	doc = astra::XMLDocument::readFromFile("test3.xml");
	BOOST_REQUIRE(doc);
	root = doc->getRootNode();
	BOOST_REQUIRE(root);
	node = root.getSingleNode("child");
	BOOST_REQUIRE(node);

	std::vector<astra::float32> f = node.getContentNumericalArray();

	BOOST_CHECK(f[0] == fl[0]);
	BOOST_CHECK(f[1] == fl[1]);
	BOOST_CHECK(f[2] == fl[2]);
	BOOST_CHECK(f[3] == fl[3]);

	delete doc;

}

BOOST_AUTO_TEST_CASE( testXMLDocument_Config )
{
	astra::Config* cfg = new astra::Config();
	cfg->initialize("VolumeGeometry2D");

	cfg->self.addChildNode("GridColCount", 1);
	cfg->self.addChildNode("GridRowCount", 2);

	cfg->self.addOption("WindowMinX", 3);
	cfg->self.addOption("WindowMaxX", 4);
	cfg->self.addOption("WindowMinY", 5);
	cfg->self.addOption("WindowMaxY", 6);

	delete cfg;
}
