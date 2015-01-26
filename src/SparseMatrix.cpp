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

#include <sstream>

#include "astra/Globals.h"
#include "astra/SparseMatrix.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// constructor

CSparseMatrix::CSparseMatrix()
{
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// constructor
CSparseMatrix::CSparseMatrix(unsigned int _iHeight, unsigned int _iWidth,
                             unsigned long _lSize)
{
	initialize(_iHeight, _iWidth, _lSize);
}


//----------------------------------------------------------------------------------------
// destructor
CSparseMatrix::~CSparseMatrix()
{
	delete[] m_pfValues;
	delete[] m_piColIndices;
	delete[] m_plRowStarts;
}

//----------------------------------------------------------------------------------------
// initialize
bool CSparseMatrix::initialize(unsigned int _iHeight, unsigned int _iWidth,
                               unsigned long _lSize)
{
	m_iHeight = _iHeight;
	m_iWidth = _iWidth;
	m_lSize = _lSize;

	m_pfValues = new float32[_lSize];
	m_piColIndices = new unsigned int[_lSize];
	m_plRowStarts = new unsigned long[_iHeight+1];
	m_bInitialized = true;

	return m_bInitialized;
}


std::string CSparseMatrix::description() const
{
	std::stringstream res;
	res << m_iHeight << "x" << m_iWidth << " sparse matrix";
	return res.str();
}




} // end namespace
