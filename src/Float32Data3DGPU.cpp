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

#include "astra/Float32Data3DGPU.h"

namespace astra {

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32Data3DGPU::CFloat32Data3DGPU()
{
	_clear();
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Destructor.
CFloat32Data3DGPU::~CFloat32Data3DGPU() 
{
	if (m_bInitialized)
	{
		_unInit();
	}
}

//----------------------------------------------------------------------------------------
// Initializes an instance of the CFloat32Data3DGPU class with pre-allocated memory
bool CFloat32Data3DGPU::_initialize(int _iWidth, int _iHeight, int _iDepth, astraCUDA3d::MemHandle3D _hnd)
{
	// basic checks
	ASTRA_ASSERT(_iWidth > 0);
	ASTRA_ASSERT(_iHeight > 0);
	ASTRA_ASSERT(_iDepth > 0);
	//ASTRA_ASSERT(_pCustomMemory != NULL);

	if (m_bInitialized) {
		_unInit();
	}

	// calculate size
	m_iWidth = _iWidth;
	m_iHeight = _iHeight;
	m_iDepth = _iDepth;
	m_iSize = (size_t)m_iWidth * m_iHeight * m_iDepth;

	m_hnd = _hnd;

	// initialization complete
	return true;
}
//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
void CFloat32Data3DGPU::_clear()
{
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDepth = 0;
	m_iSize = 0;

	m_hnd.d.reset();
}

//----------------------------------------------------------------------------------------
// Un-initialize the object, bringing it back in the unitialized state.
void CFloat32Data3DGPU::_unInit()
{
	ASTRA_ASSERT(m_bInitialized);

	_clear();
	m_bInitialized = false;
}

} // end namespace astra
