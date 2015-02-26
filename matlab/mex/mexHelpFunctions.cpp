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

/** \file mexHelpFunctions.cpp
 *
 *  \brief Contains some functions for interfacing matlab with c data structures
 */
#include "mexHelpFunctions.h"

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "astra/SparseMatrixProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/AstraObjectManager.h"

using namespace std;
using namespace astra;


//-----------------------------------------------------------------------------------------
// get string from matlab 
std::string mex_util_get_string(const mxArray* pInput)
{
	if (!mxIsChar(pInput)) {
		return "";
	}
	mwSize iLength = mxGetNumberOfElements(pInput) + 1;
	char* buf = new char[iLength]; 
	mxGetString(pInput, buf, iLength);
	std::string res = std::string(buf);
	delete[] buf;
	return res;
}

//-----------------------------------------------------------------------------------------
// is option
bool isOption(std::list<std::string> lOptions, std::string sOption) 
{
	return std::find(lOptions.begin(), lOptions.end(), sOption) != lOptions.end();
}

//-----------------------------------------------------------------------------------------
// turn a matlab struct into a c++ map
std::map<std::string, mxArray*> parseStruct(const mxArray* pInput) 
{
	std::map<std::string, mxArray*> res;

	// check type
	if (!mxIsStruct(pInput)) {
      mexErrMsgTxt("Input must be a struct.");
	  return res;
	}

	// get field names
	int nfields = mxGetNumberOfFields(pInput);
	for (int i = 0; i < nfields; i++) {
		std::string sFieldName = std::string(mxGetFieldNameByNumber(pInput, i));
		res[sFieldName] = mxGetFieldByNumber(pInput,0,i);
	}
	return res;
}

//-----------------------------------------------------------------------------------------
// turn a c++ map into a matlab struct
mxArray* buildStruct(std::map<std::string, mxArray*> mInput) 
{
	mwSize dims[2] = {1, 1};
	mxArray* res = mxCreateStructArray(2,dims,0,0);
	
	for (std::map<std::string, mxArray*>::iterator it = mInput.begin(); it != mInput.end(); it++) {
		mxAddField(res, (*it).first.c_str());
		mxSetField(res, 0, (*it).first.c_str(), (*it).second);
	}
	return res;
}

//-----------------------------------------------------------------------------------------
// parse projection geometry data
astra::CProjectionGeometry2D* parseProjectionGeometryStruct(const mxArray* prhs)
{
	// parse struct	
	std::map<string, mxArray*> mStruct = parseStruct(prhs);

	// create projection geometry object
	string type = mex_util_get_string(mStruct["type"]);
	if (type == "parallel") {

		// detector_width
		float32 fDetWidth = 1.0f;
		mxArray* tmp = mStruct["detector_width"];
		if (tmp != NULL) {
			fDetWidth = (float32)(mxGetScalar(tmp));
		}

		// detector_count
		int iDetCount = 100;
		tmp = mStruct["detector_count"];
		if (tmp != NULL) {
			iDetCount = (int)(mxGetScalar(tmp));
		}

		// angles
		float32* pfAngles;
		int iAngleCount;
		tmp = mStruct["projection_angles"];
		if (tmp != NULL) {
			double* angleValues = mxGetPr(tmp);
			iAngleCount = mxGetN(tmp) * mxGetM(tmp);
			pfAngles = new float32[iAngleCount];
			for (int i = 0; i < iAngleCount; i++) {
				pfAngles[i] = angleValues[i];
			}
		} else {
			mexErrMsgTxt("'angles' not specified, error.");
			return NULL;
		}

		// create projection geometry
		return new astra::CParallelProjectionGeometry2D(iAngleCount,	// number of projections
														iDetCount,		// number of detectors
														fDetWidth,		// width of the detectors
														pfAngles);		// angles array
	} 
	
	else if (type == "fanflat") {

		// detector_width
		float32 fDetWidth = 1.0f;
		mxArray* tmp = mStruct["detector_width"];
		if (tmp != NULL) {
			fDetWidth = (float32)(mxGetScalar(tmp));
		}

		// detector_count
		int iDetCount = 100;
		tmp = mStruct["detector_count"];
		if (tmp != NULL) {
			iDetCount = (int)(mxGetScalar(tmp));
		}

		// angles
		float32* pfAngles;
		int iAngleCount;
		tmp = mStruct["projection_angles"];
		if (tmp != NULL) {
			double* angleValues = mxGetPr(tmp);
			iAngleCount = mxGetN(tmp) * mxGetM(tmp);
			pfAngles = new float32[iAngleCount];
			for (int i = 0; i < iAngleCount; i++) {
				pfAngles[i] = angleValues[i];
			}
		} else {
			mexErrMsgTxt("'angles' not specified, error.");
			return NULL;
		}

		// origin_source_dist
		int iDistOriginSource = 100;
		tmp = mStruct["origin_source_dist"];
		if (tmp != NULL) {
			iDistOriginSource = (int)(mxGetScalar(tmp));
		}

		// origin_det_dist
		int iDistOriginDet = 100;
		tmp = mStruct["origin_det_dist"];
		if (tmp != NULL) {
			iDistOriginDet = (int)(mxGetScalar(tmp));
		}

		// create projection geometry
		return new astra::CFanFlatProjectionGeometry2D(iAngleCount,			// number of projections
													   iDetCount,			// number of detectors
													   fDetWidth,			// width of the detectors
													   pfAngles,			// angles array
													   iDistOriginSource,	// distance origin source
													   iDistOriginDet);		// distance origin detector
	}

	else {
		mexPrintf("Only parallel and fanflat projection geometry implemented.");
		return NULL;
	}
}

//-----------------------------------------------------------------------------------------
// parse reconstruction geometry data
astra::CVolumeGeometry2D* parseVolumeGeometryStruct(const mxArray* prhs)
{
	// parse struct	
	std::map<string, mxArray*> mStruct = parseStruct(prhs);

	std::map<string, mxArray*> mOptions = parseStruct(mStruct["option"]);

	// GridColCount
	int iWindowColCount = 128;
	mxArray* tmp = mStruct["GridColCount"];
	if (tmp != NULL) {
		iWindowColCount = (int)(mxGetScalar(tmp));
	} 

	// GridRowCount
	int iWindowRowCount = 128;
	tmp = mStruct["GridRowCount"];
	if (tmp != NULL) {
		iWindowRowCount = (int)(mxGetScalar(tmp));
	}

	// WindowMinX
	float32 fWindowMinX = - iWindowColCount / 2;
	tmp = mOptions["WindowMinX"];
	if (tmp != NULL) {
		fWindowMinX = (float32)(mxGetScalar(tmp));
	} 

	// WindowMaxX
	float32 fWindowMaxX = iWindowColCount / 2;
	tmp = mOptions["WindowMaxX"];
	if (tmp != NULL) {
		fWindowMaxX = (float32)(mxGetScalar(tmp));
	} 

	// WindowMinY
	float32 fWindowMinY = - iWindowRowCount / 2;
	tmp = mOptions["WindowMinY"];
	if (tmp != NULL) {
		fWindowMinY = (float32)(mxGetScalar(tmp));
	} 

	// WindowMaxX
	float32 fWindowMaxY = iWindowRowCount / 2;
	tmp = mOptions["WindowMaxY"];
	if (tmp != NULL) {
		fWindowMaxY = (float32)(mxGetScalar(tmp));
	} 
	
	// create and return reconstruction geometry
	return new astra::CVolumeGeometry2D(iWindowColCount, iWindowRowCount, 
										fWindowMinX, fWindowMinY, 
										fWindowMaxX, fWindowMaxY);
}


//-----------------------------------------------------------------------------------------
string matlab2string(const mxArray* pField)
{
	// is string?
	if (mxIsChar(pField)) {
		return mex_util_get_string(pField);
	}

	// is scalar?
	if (mxIsNumeric(pField) && mxGetM(pField)*mxGetN(pField) == 1) {
		return boost::lexical_cast<string>(mxGetScalar(pField));
	}

	return "";
}

//-----------------------------------------------------------------------------------------
// Options struct to xml node
bool readOptions(XMLNode* node, const mxArray* pOptionStruct)
{
	// loop all fields
	int nfields = mxGetNumberOfFields(pOptionStruct);
	for (int i = 0; i < nfields; i++) {
		std::string sFieldName = std::string(mxGetFieldNameByNumber(pOptionStruct, i));
		const mxArray* pField = mxGetFieldByNumber(pOptionStruct, 0, i);

		if (node->hasOption(sFieldName)) {
			mexErrMsgTxt("Duplicate option");
			return false;
		}
	
		// string or scalar
		if (mxIsChar(pField) || mex_is_scalar(pField)) {
			string sValue = matlab2string(pField);
			node->addOption(sFieldName, sValue);
		} else
		// numerical array
		if (mxIsNumeric(pField) && mxGetM(pField)*mxGetN(pField) > 1) {
			if (!mxIsDouble(pField)) {
				mexErrMsgTxt("Numeric input must be double.");
				return false;
			}

			XMLNode* listbase = node->addChildNode("Option");
			listbase->addAttribute("key", sFieldName);
			listbase->addAttribute("listsize", mxGetM(pField)*mxGetN(pField));
			double* pdValues = mxGetPr(pField);
			int index = 0;
			for (unsigned int row = 0; row < mxGetM(pField); row++) {
				for (unsigned int col = 0; col < mxGetN(pField); col++) {
					XMLNode* item = listbase->addChildNode("ListItem");
					item->addAttribute("index", index);
					item->addAttribute("value", pdValues[col*mxGetM(pField)+row]);
					index++;
					delete item;
				}
			}
			delete listbase;
		} else {
			mexErrMsgTxt("Unsupported option type");
			return false;
		}
	}
	return true;
}

//-----------------------------------------------------------------------------------------
// struct to xml node
bool readStruct(XMLNode* root, const mxArray* pStruct)
{
	// loop all fields
	int nfields = mxGetNumberOfFields(pStruct);
	for (int i = 0; i < nfields; i++) {

		// field and fieldname
		std::string sFieldName = std::string(mxGetFieldNameByNumber(pStruct, i));
		const mxArray* pField = mxGetFieldByNumber(pStruct, 0, i);

		// string
		if (mxIsChar(pField)) {
			string sValue = matlab2string(pField);
			if (sFieldName == "type") {
				root->addAttribute("type", sValue);
			} else {
				delete root->addChildNode(sFieldName, sValue);
			}
		}

		// scalar
		if (mex_is_scalar(pField)) {
			string sValue = matlab2string(pField);
			delete root->addChildNode(sFieldName, sValue);
		}

		// numerical array
		if (mxIsNumeric(pField) && mxGetM(pField)*mxGetN(pField) > 1) {
			if (!mxIsDouble(pField)) {
				mexErrMsgTxt("Numeric input must be double.");
				return false;
			}
			XMLNode* listbase = root->addChildNode(sFieldName);
			listbase->addAttribute("listsize", mxGetM(pField)*mxGetN(pField));
			double* pdValues = mxGetPr(pField);
			int index = 0;
			for (unsigned int row = 0; row < mxGetM(pField); row++) {
				for (unsigned int col = 0; col < mxGetN(pField); col++) {
					XMLNode* item = listbase->addChildNode("ListItem");
					item->addAttribute("index", index);
					item->addAttribute("value", pdValues[col*mxGetM(pField)+row]);
					index++;
					delete item;
				}
			}
			delete listbase;
		}


		// not castable to a single string
		if (mxIsStruct(pField)) {
			if (sFieldName == "options" || sFieldName == "option" || sFieldName == "Options" || sFieldName == "Option") {
				bool ret = readOptions(root, pField);
				if (!ret)
					return false;
			} else {
				XMLNode* newNode = root->addChildNode(sFieldName);
				bool ret = readStruct(newNode, pField);
				delete newNode;
				if (!ret)
					return false;
			}
		}

	}

	return true;
}

//-----------------------------------------------------------------------------------------
// turn a MATLAB struct into an XML Document
XMLDocument* struct2XML(string rootname, const mxArray* pStruct)
{
	if (!mxIsStruct(pStruct)) {
      mexErrMsgTxt("Input must be a struct.");
	  return NULL;
	}

	// create the document
	XMLDocument* doc = XMLDocument::createDocument(rootname);
	XMLNode* rootnode = doc->getRootNode();

	// read the struct
	bool ret = readStruct(rootnode, pStruct);
	delete rootnode;

	if (!ret) {
		delete doc;
		doc = 0;
	}

	return doc;
}





//-----------------------------------------------------------------------------------------
// turn an std vector<float32> object to an mxArray
mxArray* vectorToMxArray(std::vector<astra::float32> mInput)
{
	mxArray* res = mxCreateDoubleMatrix(1, mInput.size(), mxREAL);
	double* pdData = mxGetPr(res);
	for (unsigned int i = 0; i < mInput.size(); i++) {
		pdData[i] = mInput[i];
	}
	return res;
}

//-----------------------------------------------------------------------------------------
// turn a vector<vector<float32>> object to an mxArray
mxArray* vector2DToMxArray(std::vector<std::vector<astra::float32> > mInput)
{
	unsigned int sizex = mInput.size();
	if (sizex == 0) return mxCreateString("empty");
	unsigned int sizey = mInput[0].size();

	mxArray* res = mxCreateDoubleMatrix(sizex, sizey, mxREAL);
	double* pdData = mxGetPr(res);
	for (unsigned int i = 0; i < sizex; i++) {
		for (unsigned int j = 0; j < sizey && j < mInput[i].size(); j++) {
			pdData[j*sizex+i] = mInput[i][j];
		}
	}
	return res;
}

//-----------------------------------------------------------------------------------------
// turn a boost::any object to an mxArray
mxArray* anyToMxArray(boost::any _any) 
{
	if (_any.type() == typeid(std::string)) {
		std::string str = boost::any_cast<std::string>(_any);
		return mxCreateString(str.c_str());     
	}
	if (_any.type() == typeid(int)) {
		return mxCreateDoubleScalar(boost::any_cast<int>(_any));
	}
	if (_any.type() == typeid(float32)) {
		return mxCreateDoubleScalar(boost::any_cast<float32>(_any));
	}
	if (_any.type() == typeid(std::vector<astra::float32>)) {
		return vectorToMxArray(boost::any_cast<std::vector<float32> >(_any));
	}
	if (_any.type() == typeid(std::vector<std::vector<astra::float32> >)) {
		return vector2DToMxArray(boost::any_cast<std::vector<std::vector<float32> > >(_any));
	}
	return NULL;
}
//-----------------------------------------------------------------------------------------
// return true ig the argument is a scalar
bool mex_is_scalar(const mxArray* pInput)
{
	return (mxIsNumeric(pInput) && mxGetM(pInput)*mxGetN(pInput) == 1);
}














//-----------------------------------------------------------------------------------------
mxArray* configToStruct(astra::Config* cfg)
{
	return XMLNode2struct(cfg->self);
}

//-----------------------------------------------------------------------------------------
mxArray* XML2struct(astra::XMLDocument* xml)
{
	XMLNode* node = xml->getRootNode();
	mxArray* str = XMLNode2struct(xml->getRootNode());
	delete node;
	return str;
}

//-----------------------------------------------------------------------------------------
mxArray* stringToMxArray(std::string input) 
{
	// matrix
	if (input.find(';') != std::string::npos) {

		// split rows
		std::vector<std::string> row_strings;
		std::vector<std::string> col_strings;
		boost::split(row_strings, input, boost::is_any_of(";"));
		boost::split(col_strings, row_strings[0], boost::is_any_of(","));

		// get dimensions
		int rows = row_strings.size();
		int cols = col_strings.size();

		// init matrix
		mxArray* pMatrix = mxCreateDoubleMatrix(rows, cols, mxREAL);
		double* out = mxGetPr(pMatrix);

		// loop elements
		for (unsigned int row = 0; row < rows; row++) {
			boost::split(col_strings, row_strings[row], boost::is_any_of(","));
			// check size
			for (unsigned int col = 0; col < col_strings.size(); col++) {
				out[col*rows + row] = boost::lexical_cast<float32>(col_strings[col]);
			}
		}
		return pMatrix;
	}
	
	// vector
	if (input.find(',') != std::string::npos) {

		// split
		std::vector<std::string> items;
		boost::split(items, input, boost::is_any_of(","));

		// init matrix
		mxArray* pVector = mxCreateDoubleMatrix(1, items.size(), mxREAL);
		double* out = mxGetPr(pVector);

		// loop elements
		for (unsigned int i = 0; i < items.size(); i++) {
			out[i] = boost::lexical_cast<float32>(items[i]);
		}
		return pVector;
	}
	
	// number
	char* end;
	double content = ::strtod(input.c_str(), &end);
	bool isnumber = !*end;
	if (isnumber) {
		return mxCreateDoubleScalar(content);
	}
	
	// string
	return mxCreateString(input.c_str());
}

//-----------------------------------------------------------------------------------------
mxArray* XMLNode2struct(astra::XMLNode* node)
{
	std::map<std::string, mxArray*> mList;
	std::map<std::string, mxArray*> mOptions;

	// type_attribute
	if (node->hasAttribute("type")) {
		mList["type"] = mxCreateString(node->getAttribute("type").c_str());
	}

	list<XMLNode*> nodes = node->getNodes();
	for (list<XMLNode*>::iterator it = nodes.begin(); it != nodes.end(); it++) {
		XMLNode* subnode = (*it);

		// option
		if (subnode->getName() == "Option") {
			mOptions[subnode->getAttribute("key")] = stringToMxArray(subnode->getAttribute("value"));
		}

		// regular content
		else {
			mList[subnode->getName()] = stringToMxArray(subnode->getContent());
		}
		delete subnode;
	}

	if (mOptions.size() > 0) mList["options"] = buildStruct(mOptions);
	return buildStruct(mList);
}










//-----------------------------------------------------------------------------------------
void get3DMatrixDims(const mxArray* x, mwSize *dims)
{
	const mwSize* mdims = mxGetDimensions(x);
	mwSize dimCount = mxGetNumberOfDimensions(x);
	if (dimCount == 1) {
		dims[0] = mdims[0];
		dims[1] = 1;
		dims[2] = 1;
	} else if (dimCount == 2) {
		dims[0] = mdims[0];
		dims[1] = mdims[1];
		dims[2] = 1;
	} else if (dimCount == 3) {
		dims[0] = mdims[0];
		dims[1] = mdims[1];
		dims[2] = mdims[2];
	} else {
		dims[0] = 0;
		dims[1] = 0;
		dims[2] = 0;
	}
} 
