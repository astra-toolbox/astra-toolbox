/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#include "astra/XMLNode.h"

#ifdef _MSC_VER
#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"
#else
#include "rapidxml.hpp"
#include "rapidxml_print.hpp"
#endif

#include <boost/lexical_cast.hpp>

using namespace rapidxml;
using namespace astra;
using namespace std;


//-----------------------------------------------------------------------------
// Utility function to delete a list of nodes
static void deleteNodes(list<XMLNode*>& nodes)
{
	for (list<XMLNode*>::iterator i = nodes.begin(); i != nodes.end(); ++i)
		delete (*i);

	nodes.clear();
}


//-----------------------------------------------------------------------------
// default constructor
XMLNode::XMLNode() 
{

}

//-----------------------------------------------------------------------------	
// private constructor
XMLNode::XMLNode(xml_node<>* node) 
{
	fDOMElement = node;
}

//-----------------------------------------------------------------------------	
// destructor
XMLNode::~XMLNode() 
{

}

//-----------------------------------------------------------------------------	
// set DOM node (private)
void XMLNode::setDOMNode(xml_node<>* n) 
{
	fDOMElement = n;
}

//-----------------------------------------------------------------------------	
// print XML Node
void XMLNode::print()
{
	std::cout << fDOMElement;
}

//-----------------------------------------------------------------------------	
// print XML Node
std::string XMLNode::toString()
{
	std::string s;
	::print(std::back_inserter(s), *fDOMElement, 0);
	return s;
}

//-----------------------------------------------------------------------------	
// Get single node
XMLNode* XMLNode::getSingleNode(string name) 
{
	xml_node<> *node = fDOMElement->first_node(name.c_str());

	if (node)
		return new XMLNode(node);
	else
		return 0;
}

//-----------------------------------------------------------------------------	
// Get list of nodes
list<XMLNode*> XMLNode::getNodes(string name) 
{	
	list<XMLNode*> result;
	xml_node<> *iter;
	for (iter = fDOMElement->first_node(name.c_str()); iter; iter = iter->next_sibling(name.c_str())) {
		result.push_back(new XMLNode(iter));
	}
	return result;
}

//-----------------------------------------------------------------------------	
// Get list of nodes
list<XMLNode*> XMLNode::getNodes() 
{	
	list<XMLNode*> result;
	xml_node<> *iter;
	for (iter = fDOMElement->first_node(); iter; iter = iter->next_sibling()) {
		result.push_back(new XMLNode(iter));
	}
	return result;
}

//-----------------------------------------------------------------------------	
// Get name of this node
std::string XMLNode::getName()
{
	return fDOMElement->name();
}

//-----------------------------------------------------------------------------	
// Get node content - STRING
string XMLNode::getContent() 
{
	return fDOMElement->value();
}

//-----------------------------------------------------------------------------	
// Get node content - NUMERICAL
float32 XMLNode::getContentNumerical() 
{
	return boost::lexical_cast<float32>(getContent());
}

//-----------------------------------------------------------------------------	
// Get node content - BOOLEAN
bool XMLNode::getContentBool() 
{
	string res = getContent();
	return ((res == "1") || (res == "yes") || (res == "true") || (res == "on"));
}

//-----------------------------------------------------------------------------	
// Get node content - STRING LIST
vector<string> XMLNode::getContentArray()
{
	// get listsize
	int iSize = boost::lexical_cast<int>(getAttribute("listsize"));
	// create result array
	vector<string> res(iSize);
	// loop all list item nodes
	list<XMLNode*> nodes = getNodes("ListItem");
	for (list<XMLNode*>::iterator it = nodes.begin(); it != nodes.end(); it++) {
		int iIndex = (*it)->getAttributeNumerical("index");
		string sValue = (*it)->getAttribute("value");
		ASTRA_ASSERT(iIndex < iSize);
		res[iIndex] = sValue;
	}
	deleteNodes(nodes);

	// return 
	return res;
}

//-----------------------------------------------------------------------------	
// Get node content - NUMERICAL LIST
vector<float32> XMLNode::getContentNumericalArray()
{
	// is scalar
	if (!hasAttribute("listsize")) {
		vector<float32> res(1);
		res[0] = getContentNumerical();
		return res;
	}

	int iSize = boost::lexical_cast<int>(getAttribute("listsize"));
	// create result array
	vector<float32> res(iSize);
	// loop all list item nodes
	list<XMLNode*> nodes = getNodes("ListItem");
	for (list<XMLNode*>::iterator it = nodes.begin(); it != nodes.end(); it++) {
		int iIndex = (*it)->getAttributeNumerical("index");
		float32 fValue = (*it)->getAttributeNumerical("value");
		ASTRA_ASSERT(iIndex < iSize);
		res[iIndex] = fValue;
	}
	deleteNodes(nodes);
	// return 
	return res;
}

vector<double> XMLNode::getContentNumericalArrayDouble()
{
	// is scalar
	if (!hasAttribute("listsize")) {
		vector<double> res(1);
		res[0] = getContentNumerical();
		return res;
	}

	int iSize = boost::lexical_cast<int>(getAttribute("listsize"));
	// create result array
	vector<double> res(iSize);
	// loop all list item nodes
	list<XMLNode*> nodes = getNodes("ListItem");
	for (list<XMLNode*>::iterator it = nodes.begin(); it != nodes.end(); it++) {
		int iIndex = (*it)->getAttributeNumerical("index");
		double fValue = (*it)->getAttributeNumericalDouble("value");
		ASTRA_ASSERT(iIndex < iSize);
		res[iIndex] = fValue;
	}
	deleteNodes(nodes);
	// return 
	return res;
}

//-----------------------------------------------------------------------------	
// Get node content - NUMERICAL LIST 2
void XMLNode::getContentNumericalArray(float32*& _pfData, int& _iSize)
{
	// is scalar
	if (!hasAttribute("listsize")) {
		_iSize = 1;
		_pfData = new float32[_iSize];
		_pfData[0] = getContentNumerical();
		return;
	}
	// get listsize
	_iSize = boost::lexical_cast<int>(getAttribute("listsize"));
	// create result array
	_pfData = new float32[_iSize];
	// loop all list item nodes
	list<XMLNode*> nodes = getNodes("ListItem");
	for (list<XMLNode*>::iterator it = nodes.begin(); it != nodes.end(); it++) {
		int iIndex = (*it)->getAttributeNumerical("index");
		float32 fValue = (*it)->getAttributeNumerical("value");
		ASTRA_ASSERT(iIndex < _iSize);
		_pfData[iIndex] = fValue;
	}
	deleteNodes(nodes);
}

//-----------------------------------------------------------------------------	
// Is attribute?
bool XMLNode::hasAttribute(string _sName)
{
	xml_attribute<> *attr = fDOMElement->first_attribute(_sName.c_str());
	return (attr != 0);
}

//-----------------------------------------------------------------------------	
// Get attribute - STRING
string XMLNode::getAttribute(string _sName, string _sDefaultValue)
{
	xml_attribute<> *attr = fDOMElement->first_attribute(_sName.c_str());

	if (!attr) return _sDefaultValue;

	return attr->value();
}

//-----------------------------------------------------------------------------	
// Get attribute - NUMERICAL
float32 XMLNode::getAttributeNumerical(string _sName, float32 _fDefaultValue)
{
	if (!hasAttribute(_sName)) return _fDefaultValue;
	return boost::lexical_cast<float32>(getAttribute(_sName));
}
double XMLNode::getAttributeNumericalDouble(string _sName, double _fDefaultValue)
{
	if (!hasAttribute(_sName)) return _fDefaultValue;
	return boost::lexical_cast<double>(getAttribute(_sName));
}

//-----------------------------------------------------------------------------	
// Get attribute - BOOLEAN
bool XMLNode::getAttributeBool(string _sName, bool _bDefaultValue)
{
	if (!hasAttribute(_sName)) return _bDefaultValue;
	string res = getAttribute(_sName);
	return ((res == "1") || (res == "yes") || (res == "true") || (res == "on"));
}

//-----------------------------------------------------------------------------	
// Has option?
bool XMLNode::hasOption(string _sKey) 
{
	xml_node<> *iter;
	for (iter = fDOMElement->first_node("Option"); iter; iter = iter->next_sibling("Option")) {
		xml_attribute<> *attr = iter->first_attribute("key");
		if (attr && _sKey == attr->value())
			return true;
	}
	return false;
}

//-----------------------------------------------------------------------------	
// Get option - STRING
string XMLNode::getOption(string _sKey, string _sDefaultValue) 
{
	xml_node<> *iter;
	for (iter = fDOMElement->first_node("Option"); iter; iter = iter->next_sibling("Option")) {
		xml_attribute<> *attr = iter->first_attribute("key");
		if (attr && _sKey == attr->value()) {
			attr = iter->first_attribute("value");
			if (!attr)
				return "";
			return attr->value();
		}
	}
	return _sDefaultValue;
}

//-----------------------------------------------------------------------------	
// Get option - NUMERICAL
float32 XMLNode::getOptionNumerical(string _sKey, float32 _fDefaultValue) 
{
	if (!hasOption(_sKey)) return _fDefaultValue;
	return boost::lexical_cast<float32>(getOption(_sKey));
}

//-----------------------------------------------------------------------------	
// Get option - BOOL
bool XMLNode::getOptionBool(string _sKey, bool _bDefaultValue)
{
	bool bHasOption = hasOption(_sKey);
	if (!bHasOption) return _bDefaultValue;
	string res = getOption(_sKey);
	return ((res == "1") || (res == "yes") || (res == "true") || (res == "on"));
}

//-----------------------------------------------------------------------------	
// Get option - NUMERICAL ARRAY
vector<float32> XMLNode::getOptionNumericalArray(string _sKey)
{
	if (!hasOption(_sKey)) return vector<float32>();

	list<XMLNode*> nodes = getNodes("Option");
	for (list<XMLNode*>::iterator it = nodes.begin(); it != nodes.end(); it++) {
		if ((*it)->getAttribute("key") == _sKey) {
			vector<float32> vals = (*it)->getContentNumericalArray();
			deleteNodes(nodes);
			return vals;
		}
	}

	deleteNodes(nodes);
	return vector<float32>();
}

//-----------------------------------------------------------------------------	












//-----------------------------------------------------------------------------	
// BUILD NODE
//-----------------------------------------------------------------------------	

//-----------------------------------------------------------------------------
// Add child node - EMPTY
XMLNode* XMLNode::addChildNode(string _sNodeName) 
{
	xml_document<> *doc = fDOMElement->document();
	char *node_name = doc->allocate_string(_sNodeName.c_str());
	xml_node<> *node = doc->allocate_node(node_element, node_name);
	fDOMElement->append_node(node);

	// TODO: clean up: this 'new' requires callers to do memory management
	return new XMLNode(node);
}

//-----------------------------------------------------------------------------
// Add child node - STRING
XMLNode* XMLNode::addChildNode(string _sNodeName, string _sText) 
{
	XMLNode* res = addChildNode(_sNodeName);
	res->setContent(_sText);
	return res;
}

//-----------------------------------------------------------------------------
// Add child node - FLOAT
XMLNode* XMLNode::addChildNode(string _sNodeName, float32 _fValue) 
{
	XMLNode* res = addChildNode(_sNodeName);
	res->setContent(_fValue);
	return res;
}

//-----------------------------------------------------------------------------
// Add child node - LIST
XMLNode* XMLNode::addChildNode(string _sNodeName, float32* _pfList, int _iSize) 
{
	XMLNode* res = addChildNode(_sNodeName);
	res->setContent(_pfList, _iSize);
	return res;
}

//-----------------------------------------------------------------------------	
// Set content - STRING
void XMLNode::setContent(string _sText) 
{
	xml_document<> *doc = fDOMElement->document();
	char *text = doc->allocate_string(_sText.c_str());
	fDOMElement->value(text);
}

//-----------------------------------------------------------------------------	
// Set content - FLOAT
void XMLNode::setContent(float32 _fValue) 
{
	setContent(boost::lexical_cast<string>(_fValue));
}

//-----------------------------------------------------------------------------	
// Set content - LIST
void XMLNode::setContent(float32* pfList, int _iSize) 
{
	std::string str = (_iSize > 0) ? boost::lexical_cast<std::string>(pfList[0]) : "";
	for (int i = 1; i < _iSize; i++) {
		str += "," + boost::lexical_cast<std::string>(pfList[i]);
	}
	setContent(str);
}

//-----------------------------------------------------------------------------	
// Add attribute - STRING
void XMLNode::addAttribute(string _sName, string _sText) 
{
	xml_document<> *doc = fDOMElement->document();
	char *name = doc->allocate_string(_sName.c_str());
	char *text = doc->allocate_string(_sText.c_str());
	xml_attribute<> *attr = doc->allocate_attribute(name, text);
	fDOMElement->append_attribute(attr);
}

//-----------------------------------------------------------------------------	
// Add attribute - FLOAT
void XMLNode::addAttribute(string _sName, float32 _fValue) 
{
	addAttribute(_sName, boost::lexical_cast<string>(_fValue)); 
}

//-----------------------------------------------------------------------------	
// Add option - STRING
void XMLNode::addOption(string _sName, string _sText) 
{
	XMLNode* node = addChildNode("Option");
	node->addAttribute("key", _sName);
	node->addAttribute("value", _sText);
	delete node;
}

//-----------------------------------------------------------------------------	
// Add option - FLOAT
void XMLNode::addOption(string _sName, float32 _sText) 
{
	XMLNode* node = addChildNode("Option");
	node->addAttribute("key", _sName);
	node->addAttribute("value", _sText);
	delete node;
}
//-----------------------------------------------------------------------------	


