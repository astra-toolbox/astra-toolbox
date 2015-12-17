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

#ifndef _INC_ASTRA_XMLNODE
#define _INC_ASTRA_XMLNODE

#include <list>
#include <string>
#include <vector>

#if 1
namespace rapidxml {
    template<class Ch> class xml_node;
}
#else
#include "rapidxml.hpp"
#endif

#include "Globals.h"
#include "Utilities.h"

using namespace std;

namespace astra {

/**
 * This class encapsulates an XML Node of the Xerces DOM Parser.
 */
class _AstraExport XMLNode {

friend class XMLDocument;

public:
	
	/** Default Constructor 
	 */
	XMLNode();

	/** Deconstructor
	 */
	~XMLNode();

	/** Check validity
	 */
	operator bool() const { return fDOMElement != 0; }

	/** Get a single child XML node. If there are more, the first one is returned
	 *
	 * @param _sName tagname of the requested child node
	 * @return first child node with the correct tagname, null pointer if it doesn't exist
	 */
	XMLNode getSingleNode(string _sName) const;

	/** Get all child XML nodes that have the tagname name
	 * 
	 * @param _sName tagname of the requested child nodes
	 * @return list with all child nodes with the correct tagname
	 */
	std::list<XMLNode> getNodes(string _sName) const;
	
	/** Get all child XML nodes
	 * 
	 * @return list with all child nodes 
	 */
	std::list<XMLNode> getNodes() const;

	/** Get the name of this node
	 * 
	 * @return name of node
	 */
	std::string getName() const;

	/** Get the content of the XML node as a single string.
	 *
	 * @return node content
	 */ 
	string getContent() const;

	/** Get the content of the XML node as an integer
	 *
	 * @return node content
	 */
	int getContentInt() const;

	/** Get the content of the XML node as a numerical.
	 *
	 * @return node content
	 */ 
	float32 getContentNumerical() const;

	/** Get the content of the XML node as a boolean.
	 *
	 * @return node content
	 */ 
	bool getContentBool() const;

	/** Get the content of the XML node as a vector of strings.
	 *
	 * @return node content
	 */ 
	vector<string> getContentArray() const;

	/** Get the content of the XML node as a stl container of float32 data.
	 *  NB: A 2D matrix is returned as a linear list
	 *
	 * @return node content
	 */ 
	vector<float32> getContentNumericalArray() const;
	vector<double> getContentNumericalArrayDouble() const;



	/** Does this node contain an attribute with a certain name?
	 *
	 * @param _sName of the attribute.
	 * @return attribute value, empty string if it doesn't exist.
	 */ 
	bool hasAttribute(string _sName) const;

	/** Get the value of an attribute.
	 *
	 * @param _sName of the attribute.
	 * @param _sDefaultValue value to return if the attribute isn't found
	 * @return attribute value, _sDefaultValue if it doesn't exist.
	 */ 
	string getAttribute(string _sName, string _sDefaultValue = "") const;

	/** Get the value of a numerical attribute.
	 *
	 * @param _sName of the attribute.
	 * @param _fDefaultValue value to return if the attribute isn't found
	 * @return attribute value, _fDefaultValue if it doesn't exist.
	 */ 
	float32 getAttributeNumerical(string _sName, float32 _fDefaultValue = 0) const;
	double getAttributeNumericalDouble(string _sName, double _fDefaultValue = 0) const;
	int getAttributeInt(string _sName, int _fDefaultValue = 0) const;

	/** Get the value of a boolean attribute.
	 *
	 * @param _sName of the attribute.
	 * @param _bDefaultValue value to return if the attribute isn't found
	 * @return attribute value, _bDefaultValue if it doesn't exist.
	 */ 
	bool getAttributeBool(string _sName, bool _bDefaultValue = false) const;




	/** Does this node contain an option with a certain key?
	 *
	 * @param _sKey option key
	 * @return true if option does exist
	 */ 
	bool hasOption(string _sKey) const;

	/** Get the value of an option within this XML Node
	 *
	 * @param _sKey option key
	 * @param _sDefaultValue value to return if key isn't found
	 * @return option value, _sDefaultValue if the option doesn't exist
	 */ 
	string getOption(string _sKey, string _sDefaultValue = "") const;

	/** Get the value of an option within this XML Node
	 *
	 * @param _sKey option key
	 * @param _fDefaultValue value to return if key isn't found
	 * @return option value, _fDefaultValue if the option doesn't exist
	 */ 
	float32 getOptionNumerical(string _sKey, float32 _fDefaultValue = 0) const;
	int getOptionInt(string _sKey, int _fDefaultValue = 0) const;

	/** Get the value of an option within this XML Node
	 *
	 * @param _sKey option key
	 * @param _bDefaultValue value to return if key isn't found
	 * @return option value, _bDefaultValue if the option doesn't exist
	 */ 
	bool getOptionBool(string _sKey, bool _bDefaultValue = false) const;

	/** Get the value of an option within this XML Node
	 *
	 * @param _sKey option key
	 * @return numerical array
	 */ 
	vector<float32> getOptionNumericalArray(string _sKey) const;





	/** Create a new XML node as a child to this one: &lt;...&gt;&lt;_sNodeName/&gt;</...&gt;
	 *
	 * @param _sNodeName the name of the new childnode
	 * @return new child node
	 */ 
	XMLNode addChildNode(string _sNodeName);

	/** Create a new XML node as a child to this one, also add some content: 
	 * &lt;...&gt;&lt;_sNodeName&gt;_sValue&lt;/_sNodeName>&lt;/...&gt;
	 *
	 * @param _sNodeName the name of the new childnode
	 * @param _sValue some node content
	 * @return new child node
	 */ 
	XMLNode addChildNode(string _sNodeName, string _sValue);

	/** Create a new XML node as a child to this one, also add some numerical content: 
	 * &lt;...&gt;&lt;_sNodeName&gt;_sValue&lt;/_sNodeName>&lt;/...&gt;
	 *
	 * @param _sNodeName the name of the new childnode
	 * @param _fValue some node content
	 * @return new child node
	 */ 
	XMLNode addChildNode(string _sNodeName, float32 _fValue);

	/** Create a new XML node as a child to this one, also add a list of numerical content: 
	 * &lt;...&gt;&lt;_sNodeName&gt;_sValue&lt;/_sNodeName>&lt;/...&gt;
	 *
	 * @param _sNodeName the name of the new childnode
	 * @param _pfList list data
	 * @param _iSize number of elements in _pfList
	 * @return new child node
	 */ 
	XMLNode addChildNode(string _sNodeName, float32* _pfList, int _iSize);

	/** Add some text to the node: &lt;...&gt;_sText&lt;/...&gt;
	 *
	 * @param _sText text to insert
	 */ 
	void setContent(string _sText);

	/** Add a number to the node: &lt;...&gt;_sText&lt;/...&gt;
	 *
	 * @param _fValue number to insert
	 */ 
	void setContent(float32 _fValue);

	/** Add a list of numerical data to the node
	 *
	 * @param _pfList data
	 * @param _iSize number of elements in the list
	 */ 
	void setContent(float32* _pfList, int _iSize);

	/** Add a list of numerical data to the node
	 *
	 * @param _pfList data
	 * @param _iSize number of elements in the list
	 */
	void setContent(double* _pfList, int _iSize);

	/** Add a (2D) matrix of numerical data to the node
	 *
	 * @param _pfMatrix data
	 * @param _iWidth width of the matrix
	 * @param _iHeight height of the matrix
	 * @param transposed true is C order, false is Fortran order
	 */
	void setContent(float32* _pfMatrix, int _iWidth, int _iHeight, bool transposed);

	/** Add a (2D) matrix of numerical data to the node
	 *
	 * @param _pfMatrix data
	 * @param _iWidth width of the matrix
	 * @param _iHeight height of the matrix
	 * @param transposed true is C order, false is Fortran order
	 */
	void setContent(double* _pfMatrix, int _iWidth, int _iHeight, bool transposed);

	/** Add an attribute to this node: &lt;... _sName="_sValue"&gt;
	 *
	 * @param _sName name of the attribute
	 * @param _sValue value of the attribute
	 */ 
	void addAttribute(string _sName, string _sValue);	

	/** Add an attribute with numerical data to this node: &lt;... _sName="_fValue"&gt;
	 *
	 * @param _sName name of the attribute
	 * @param _sValue value of the attribute
	 */ 
	void addAttribute(string _sName, float32 _fValue);	

	/** Add an option node as a child: &lt;Option key="&lt;_sKey&gt;" value="&lt;_sValue&gt;"/>
	 *
	 * @param _sKey option key
	 * @param _sValue option value
	 */ 
	void addOption(string _sKey, string _sValue);	

	/** Add an option node as a child: &lt;Option key="&lt;_sKey&gt;" value="&lt;_sValue&gt;"/>
	 *
	 * @param _sKey option key
	 * @param _sValue option value
	 */ 
	void addOption(string _sKey, float32 _fValue);	

	
	/** Print to String
	 */
	std::string toString() const;

	/** Print the node
	 */ 
	void print() const;

protected:

	/** Private Constructor.
	 * 
	 * @param n rapidxml node
	 */
	XMLNode(rapidxml::xml_node<char>* n);	

	/** Link this object to a rapidxml node
	 * @param n object of the Xerces C++ library  
	 */ 
	void setDOMNode(rapidxml::xml_node<char>* n);

	// todo: rename "fDOMElement" to "m_fDOMElement"?

	//!< Node of rapidxml
	rapidxml::xml_node<char>* fDOMElement; 

};

} // end namespace

#endif
