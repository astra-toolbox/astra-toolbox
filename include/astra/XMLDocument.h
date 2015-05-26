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

#ifndef _INC_ASTRA_XMLDOCUMENT
#define _INC_ASTRA_XMLDOCUMENT

#include <string>

#if 1
namespace rapidxml {
    template<class Ch> class xml_document;
}
#else
#include "rapidxml.hpp"
#endif

#include "Globals.h"
#include "XMLNode.h"

using namespace std;

namespace astra {

/** This class encapsulates an XML Document of the Xerces DOM Parser.
 */
class _AstraExport XMLDocument {
	
public:
	
	/** Default Constructor 
	 */
	XMLDocument();

	/** Destructor 
	 */
	~XMLDocument();
	
	/** Construct an XML DOM tree and Document from an XML file 
	 *
	 * @param sFilename Location of the XML file.
	 * @return XML Document containing the DOM tree
	 */
	static XMLDocument* readFromFile(string sFilename);

	/** Construct an empty XML DOM tree with a specific root tag.
	 *
	 * @param sRootName Element name of the root tag.
	 * @return XML Document with an empty root node
	 */
	static XMLDocument* createDocument(string sRootName);

	/** Get the rootnode of the XML document
	 *
	 * @return first XML node of the document
	 */
	XMLNode getRootNode();

	/** Save an XML DOM tree to an XML file
	 *
	 * @param sFilename Location of the XML file.
	 */
	void saveToFile(string sFilename);

	/** convert and XML DOM tree to a string
	 */
	std::string toString();

private:
	
	//!< Document of rapidxml
	rapidxml::xml_document<char>* fDOMDocument;

	std::string fBuf;

};

} // end namespace

#endif
