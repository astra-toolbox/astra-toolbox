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

#include "rapidxml/rapidxml.hpp"
#include "rapidxml/rapidxml_print.hpp"


using namespace rapidxml;
using namespace astra;
using namespace std;


//-----------------------------------------------------------------------------
// default constructor
XMLNode::XMLNode() 
{
	fDOMElement = 0;
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
void XMLNode::print() const
{
	std::cout << fDOMElement;
}

//-----------------------------------------------------------------------------	
// print XML Node
std::string XMLNode::toString() const
{
	std::string s;
	::print(std::back_inserter(s), *fDOMElement, 0);
	return s;
}

//-----------------------------------------------------------------------------	
// Get single node
XMLNode XMLNode::getSingleNode(string name) const
{
	xml_node<> *node = fDOMElement->first_node(name.c_str());

	return XMLNode(node);
}

//-----------------------------------------------------------------------------	
// Get list of nodes
list<XMLNode> XMLNode::getNodes(string name) const
{	
	list<XMLNode> result;
	xml_node<> *iter;
	for (iter = fDOMElement->first_node(name.c_str()); iter; iter = iter->next_sibling(name.c_str())) {
		result.push_back(XMLNode(iter));
	}
	return result;
}

//-----------------------------------------------------------------------------	
// Get list of nodes
list<XMLNode> XMLNode::getNodes() const
{	
	list<XMLNode> result;
	xml_node<> *iter;
	for (iter = fDOMElement->first_node(); iter; iter = iter->next_sibling()) {
		result.push_back(XMLNode(iter));
	}
	return result;
}

//-----------------------------------------------------------------------------	
// Get name of this node
std::string XMLNode::getName() const
{
	return fDOMElement->name();
}

//-----------------------------------------------------------------------------	
// Get node content - STRING
string XMLNode::getContent() const
{
	return fDOMElement->value();
}

//-----------------------------------------------------------------------------	
// Get node content - NUMERICAL
float32 XMLNode::getContentNumerical() const
{
	return StringUtil::stringToFloat(getContent());
}
int XMLNode::getContentInt() const
{
	return StringUtil::stringToInt(getContent());
}


//-----------------------------------------------------------------------------	
// Get node content - BOOLEAN
bool XMLNode::getContentBool() const
{
	string res = getContent();
	return ((res == "1") || (res == "yes") || (res == "true") || (res == "on"));
}

//-----------------------------------------------------------------------------	
// Get node content - STRING LIST
vector<string> XMLNode::getContentArray() const
{
	// get listsize
	int iSize = StringUtil::stringToInt(getAttribute("listsize"));
	// create result array
	vector<string> res(iSize);
	// loop all list item nodes
	list<XMLNode> nodes = getNodes("ListItem");
	for (list<XMLNode>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
		int iIndex = it->getAttributeNumerical("index");
		string sValue = it->getAttribute("value");
		ASTRA_ASSERT(iIndex < iSize);
		res[iIndex] = sValue;
	}

	// return 
	return res;
}

//-----------------------------------------------------------------------------	
// Get node content - NUMERICAL LIST
// NB: A 2D matrix is returned as a linear list
vector<float32> XMLNode::getContentNumericalArray() const
{
	return StringUtil::stringToFloatVector(getContent());
}

vector<double> XMLNode::getContentNumericalArrayDouble() const
{
	return StringUtil::stringToDoubleVector(getContent());
}

//-----------------------------------------------------------------------------	
// Is attribute?
bool XMLNode::hasAttribute(string _sName) const
{
	xml_attribute<> *attr = fDOMElement->first_attribute(_sName.c_str());
	return (attr != 0);
}

//-----------------------------------------------------------------------------	
// Get attribute - STRING
string XMLNode::getAttribute(string _sName, string _sDefaultValue) const
{
	xml_attribute<> *attr = fDOMElement->first_attribute(_sName.c_str());

	if (!attr) return _sDefaultValue;

	return attr->value();
}

//-----------------------------------------------------------------------------	
// Get attribute - NUMERICAL
float32 XMLNode::getAttributeNumerical(string _sName, float32 _fDefaultValue) const
{
	if (!hasAttribute(_sName)) return _fDefaultValue;
	return StringUtil::stringToFloat(getAttribute(_sName));
}
double XMLNode::getAttributeNumericalDouble(string _sName, double _fDefaultValue) const
{
	if (!hasAttribute(_sName)) return _fDefaultValue;
	return StringUtil::stringToDouble(getAttribute(_sName));
}
int XMLNode::getAttributeInt(string _sName, int _iDefaultValue) const
{
	if (!hasAttribute(_sName)) return _iDefaultValue;
	return StringUtil::stringToInt(getAttribute(_sName));
}


//-----------------------------------------------------------------------------	
// Get attribute - BOOLEAN
bool XMLNode::getAttributeBool(string _sName, bool _bDefaultValue) const
{
	if (!hasAttribute(_sName)) return _bDefaultValue;
	string res = getAttribute(_sName);
	return ((res == "1") || (res == "yes") || (res == "true") || (res == "on"));
}

//-----------------------------------------------------------------------------	
// Has option?
bool XMLNode::hasOption(string _sKey) const
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
string XMLNode::getOption(string _sKey, string _sDefaultValue) const
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
float32 XMLNode::getOptionNumerical(string _sKey, float32 _fDefaultValue) const
{
	if (!hasOption(_sKey)) return _fDefaultValue;
	return StringUtil::stringToFloat(getOption(_sKey));
}
int XMLNode::getOptionInt(string _sKey, int _iDefaultValue) const
{
	if (!hasOption(_sKey)) return _iDefaultValue;
	return StringUtil::stringToInt(getOption(_sKey));
}


//-----------------------------------------------------------------------------	
// Get option - BOOL
bool XMLNode::getOptionBool(string _sKey, bool _bDefaultValue) const
{
	bool bHasOption = hasOption(_sKey);
	if (!bHasOption) return _bDefaultValue;
	string res = getOption(_sKey);
	return ((res == "1") || (res == "yes") || (res == "true") || (res == "on"));
}

//-----------------------------------------------------------------------------	
// Get option - NUMERICAL ARRAY
vector<float32> XMLNode::getOptionNumericalArray(string _sKey) const
{
	if (!hasOption(_sKey)) return vector<float32>();

	list<XMLNode> nodes = getNodes("Option");
	for (list<XMLNode>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
		if (it->getAttribute("key") == _sKey) {
			vector<float32> vals = it->getContentNumericalArray();
			return vals;
		}
	}

	return vector<float32>();
}

//-----------------------------------------------------------------------------	












//-----------------------------------------------------------------------------	
// BUILD NODE
//-----------------------------------------------------------------------------	

//-----------------------------------------------------------------------------
// Add child node - EMPTY
XMLNode XMLNode::addChildNode(string _sNodeName) 
{
	xml_document<> *doc = fDOMElement->document();
	char *node_name = doc->allocate_string(_sNodeName.c_str());
	xml_node<> *node = doc->allocate_node(node_element, node_name);
	fDOMElement->append_node(node);

	return XMLNode(node);
}

//-----------------------------------------------------------------------------
// Add child node - STRING
XMLNode XMLNode::addChildNode(string _sNodeName, string _sText) 
{
	XMLNode res = addChildNode(_sNodeName);
	res.setContent(_sText);
	return res;
}

//-----------------------------------------------------------------------------
// Add child node - FLOAT
XMLNode XMLNode::addChildNode(string _sNodeName, float32 _fValue) 
{
	XMLNode res = addChildNode(_sNodeName);
	res.setContent(_fValue);
	return res;
}

//-----------------------------------------------------------------------------
// Add child node - LIST
XMLNode XMLNode::addChildNode(string _sNodeName, float32* _pfList, int _iSize) 
{
	XMLNode res = addChildNode(_sNodeName);
	res.setContent(_pfList, _iSize);
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
	setContent(StringUtil::floatToString(_fValue));
}

//-----------------------------------------------------------------------------	
// Set content - LIST

template<typename T>
static std::string setContentList_internal(T* pfList, int _iSize) {
	std::string str = (_iSize > 0) ? StringUtil::toString(pfList[0]) : "";
	for (int i = 1; i < _iSize; i++) {
		str += "," + StringUtil::toString(pfList[i]);
	}
	return str;
}

void XMLNode::setContent(float32* pfList, int _iSize)
{
	setContent(setContentList_internal<float32>(pfList, _iSize));
}

void XMLNode::setContent(double* pfList, int _iSize)
{
	setContent(setContentList_internal<double>(pfList, _iSize));
}

//-----------------------------------------------------------------------------	
// Set content - MATRIX

template<typename T>
static std::string setContentMatrix_internal(T* _pfMatrix, int _iWidth, int _iHeight, bool transposed)
{
	std::string str = "";

	int s1,s2;

	if (!transposed) {
		s1 = 1;
		s2 = _iWidth;
	} else {
		s1 = _iHeight;
		s2 = 1;
	}

	for (int y = 0; y < _iHeight; ++y) {
		if (_iWidth > 0)
			str += StringUtil::toString(_pfMatrix[0*s1 + y*s2]);
			for (int x = 1; x < _iWidth; x++)
				str += "," + StringUtil::toString(_pfMatrix[x*s1 + y*s2]);

		if (y != _iHeight-1)
			str += ";";
	}

	return str;
}

void XMLNode::setContent(float32* _pfMatrix, int _iWidth, int _iHeight, bool transposed)
{
	setContent(setContentMatrix_internal<float32>(_pfMatrix, _iWidth, _iHeight, transposed));
}

void XMLNode::setContent(double* _pfMatrix, int _iWidth, int _iHeight, bool transposed)
{
	setContent(setContentMatrix_internal<double>(_pfMatrix, _iWidth, _iHeight, transposed));
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
	addAttribute(_sName, StringUtil::floatToString(_fValue));
}

//-----------------------------------------------------------------------------	
// Add option - STRING
void XMLNode::addOption(string _sName, string _sText) 
{
	XMLNode node = addChildNode("Option");
	node.addAttribute("key", _sName);
	node.addAttribute("value", _sText);
}

//-----------------------------------------------------------------------------	
// Add option - FLOAT
void XMLNode::addOption(string _sName, float32 _sText) 
{
	XMLNode node = addChildNode("Option");
	node.addAttribute("key", _sName);
	node.addAttribute("value", _sText);
}
//-----------------------------------------------------------------------------	


