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

#include "astra/Float32VolumeData3DGPU.h"

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor
CFloat32VolumeData3DGPU::CFloat32VolumeData3DGPU() :
	CFloat32Data3DGPU() 
{
	m_pGeometry = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Create an instance of the CFloat32VolumeData2D class with pre-allocated data
CFloat32VolumeData3DGPU::CFloat32VolumeData3DGPU(CVolumeGeometry3D* _pGeometry, astraCUDA3d::MemHandle3D _hnd)
{
	m_bInitialized = false;
	m_bInitialized = initialize(_pGeometry, _hnd);
}


//----------------------------------------------------------------------------------------
// Destructor
CFloat32VolumeData3DGPU::~CFloat32VolumeData3DGPU() 
{
	delete m_pGeometry;
	m_pGeometry = 0;
}

//----------------------------------------------------------------------------------------
// Initialization
bool CFloat32VolumeData3DGPU::initialize(CVolumeGeometry3D* _pGeometry, astraCUDA3d::MemHandle3D _hnd)
{
	m_pGeometry = _pGeometry->clone();
	m_bInitialized = _initialize(m_pGeometry->getGridColCount(), m_pGeometry->getGridRowCount(), m_pGeometry->getGridSliceCount(), _hnd);
	return m_bInitialized;
}

} // end namespace astra
