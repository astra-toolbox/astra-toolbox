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

#include "astra/Float32Data3D.h"
#include <sstream>

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32Data3D::CFloat32Data3D()
{
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor. 
CFloat32Data3D::~CFloat32Data3D() 
{

}
//----------------------------------------------------------------------------------------

bool CFloat32Data3D::_data3DSizesEqual(const CFloat32Data3D * _pA, const CFloat32Data3D * _pB)
{
	return ((_pA->m_iWidth == _pB->m_iWidth) && (_pA->m_iHeight == _pB->m_iHeight) && (_pA->m_iDepth == _pB->m_iDepth));
}

std::string CFloat32Data3D::description() const
{
	std::stringstream res;
	res << m_iWidth << "x" << m_iHeight << "x" << m_iDepth;
	if (getType() == CFloat32Data3D::PROJECTION) res << " sinogram data \t";
	if (getType() == CFloat32Data3D::VOLUME) res << " volume data \t";
	return res.str();
}
//----------------------------------------------------------------------------------------


} // end namespace astra
