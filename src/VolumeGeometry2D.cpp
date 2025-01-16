/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

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
*/

#include "astra/VolumeGeometry2D.h"

#include "astra/Logging.h"
#include "astra/XMLConfig.h"

#include <cmath>

namespace astra
{

//----------------------------------------------------------------------------------------
// Check all variable values
bool CVolumeGeometry2D::_check()
{
	ASTRA_CONFIG_CHECK(m_iGridColCount > 0, "VolumeGeometry2D", "GridColCount must be strictly positive.");
	ASTRA_CONFIG_CHECK(m_iGridRowCount > 0, "VolumeGeometry2D", "GridRowCount must be strictly positive.");
	ASTRA_CONFIG_CHECK(m_fWindowMinX < m_fWindowMaxX, "VolumeGeometry2D", "WindowMinX should be lower than WindowMaxX.");
	ASTRA_CONFIG_CHECK(m_fWindowMinY < m_fWindowMaxY, "VolumeGeometry2D", "WindowMinY should be lower than WindowMaxY.");

	ASTRA_CONFIG_CHECK(fabsf(m_iGridTotCount - (m_iGridColCount * m_iGridRowCount)) < eps, "VolumeGeometry2D", "Internal configuration error (m_iGridTotCount).");
	ASTRA_CONFIG_CHECK(fabsf(m_fWindowLengthX - (m_fWindowMaxX - m_fWindowMinX)) < eps, "VolumeGeometry2D", "Internal configuration error (m_fWindowLengthX).");
	ASTRA_CONFIG_CHECK(fabsf(m_fWindowLengthY - (m_fWindowMaxY - m_fWindowMinY)) < eps, "VolumeGeometry2D", "Internal configuration error (m_fWindowLengthY).");
	ASTRA_CONFIG_CHECK(fabsf(m_fWindowArea - (m_fWindowLengthX * m_fWindowLengthY)) < eps, "VolumeGeometry2D", "Internal configuration error (m_fWindowArea).");
	ASTRA_CONFIG_CHECK(fabsf(m_fPixelLengthX - (m_fWindowLengthX / (float32)m_iGridColCount)) < eps, "VolumeGeometry2D", "Internal configuration error (m_fPixelLengthX).");
	ASTRA_CONFIG_CHECK(fabsf(m_fPixelLengthY - (m_fWindowLengthY / (float32)m_iGridRowCount)) < eps, "VolumeGeometry2D", "Internal configuration error (m_fPixelLengthY).");

	ASTRA_CONFIG_CHECK(fabsf(m_fPixelArea - (m_fPixelLengthX * m_fPixelLengthY)) < eps, "VolumeGeometry2D", "Internal configuration error (m_fPixelArea).");
	ASTRA_CONFIG_CHECK(fabsf(m_fDivPixelLengthX * m_fPixelLengthX - 1.0f) < eps, "VolumeGeometry2D", "Internal configuration error (m_fDivPixelLengthX).");
	ASTRA_CONFIG_CHECK(fabsf(m_fDivPixelLengthY * m_fPixelLengthY - 1.0f) < eps, "VolumeGeometry2D", "Internal configuration error (m_fDivPixelLengthY).");

	return true;
}

//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
void CVolumeGeometry2D::clear()
{
	m_iGridColCount = 0;
	m_iGridRowCount = 0;
	m_iGridTotCount = 0;

	m_fWindowLengthX = 0.0f;
	m_fWindowLengthY = 0.0f;
	m_fWindowArea = 0.0f;

	m_fPixelLengthX = 0.0f;
	m_fPixelLengthY = 0.0f;
	m_fPixelArea = 0.0f;

	m_fDivPixelLengthX = 0.0f;  
	m_fDivPixelLengthY = 0.0f;

	m_fWindowMinX = 0.0f;
	m_fWindowMinY = 0.0f;
	m_fWindowMaxX = 0.0f;
	m_fWindowMaxY = 0.0f;

	m_bInitialized = false;
}
	
//----------------------------------------------------------------------------------------
// Default constructor.
CVolumeGeometry2D::CVolumeGeometry2D() : configCheckData(0)
{
	clear();
}

//----------------------------------------------------------------------------------------
// Default constructor
CVolumeGeometry2D::CVolumeGeometry2D(int _iGridColCount, int _iGridRowCount)
 : configCheckData(0)
{
	clear();
	initialize(_iGridColCount, _iGridRowCount);
}

//----------------------------------------------------------------------------------------
// Constructor.
CVolumeGeometry2D::CVolumeGeometry2D(int _iGridColCount, 
									 int _iGridRowCount, 
									 float32 _fWindowMinX, 
									 float32 _fWindowMinY, 
									 float32 _fWindowMaxX, 
									 float32 _fWindowMaxY)
{
	clear();
	initialize(_iGridColCount, 
			   _iGridRowCount, 
			   _fWindowMinX, 
			   _fWindowMinY, 
			   _fWindowMaxX,
			   _fWindowMaxY);
}

//----------------------------------------------------------------------------------------
// Destructor.
CVolumeGeometry2D::~CVolumeGeometry2D()
{
	if (m_bInitialized) {
		clear();
	}
}

//----------------------------------------------------------------------------------------
// Clone
CVolumeGeometry2D* CVolumeGeometry2D::clone() const
{
	CVolumeGeometry2D* res = new CVolumeGeometry2D();
	res->m_bInitialized		= m_bInitialized;
	res->m_iGridColCount	= m_iGridColCount;
	res->m_iGridRowCount	= m_iGridRowCount;
	res->m_iGridTotCount	= m_iGridTotCount;
	res->m_fWindowLengthX	= m_fWindowLengthX;
	res->m_fWindowLengthY	= m_fWindowLengthY;
	res->m_fWindowArea		= m_fWindowArea;
	res->m_fPixelLengthX	= m_fPixelLengthX;
	res->m_fPixelLengthY	= m_fPixelLengthY;
	res->m_fPixelArea		= m_fPixelArea;
	res->m_fDivPixelLengthX = m_fDivPixelLengthX;
	res->m_fDivPixelLengthY = m_fDivPixelLengthY;
	res->m_fWindowMinX		= m_fWindowMinX;
	res->m_fWindowMinY		= m_fWindowMinY;
	res->m_fWindowMaxX		= m_fWindowMaxX;
	res->m_fWindowMaxY		= m_fWindowMaxY;
	return res;
}

//----------------------------------------------------------------------------------------
// Initialization witha Config object
bool CVolumeGeometry2D::initialize(const Config& _cfg)
{
	ConfigReader<CVolumeGeometry2D> CR("VolumeGeometry2D", this, _cfg);
	
	// uninitialize if the object was initialized before
	if (m_bInitialized)	{
		clear();
	}

	bool ok = true;

	// Required: number of voxels
	ok &= CR.getRequiredInt("GridColCount", m_iGridColCount);
	ok &= CR.getRequiredInt("GridRowCount", m_iGridRowCount);

	// Optional: window minima and maxima
	ok &= CR.getOptionNumerical("WindowMinX", m_fWindowMinX, -m_iGridColCount/2.0f);
	ok &= CR.getOptionNumerical("WindowMaxX", m_fWindowMaxX, m_iGridColCount/2.0f);
	ok &= CR.getOptionNumerical("WindowMinY", m_fWindowMinY, -m_iGridRowCount/2.0f);
	ok &= CR.getOptionNumerical("WindowMaxY", m_fWindowMaxY, m_iGridRowCount/2.0f);

	if (!ok)
		return false;

	_calculateDependents();

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CVolumeGeometry2D::initialize(int _iGridColCount, int _iGridRowCount)
{
	return initialize(_iGridColCount, 
					  _iGridRowCount, 
					  -_iGridColCount/2.0f,
					  -_iGridRowCount/2.0f,
					  _iGridColCount/2.0f,
					  _iGridRowCount/2.0f);
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CVolumeGeometry2D::initialize(int _iGridColCount, 
								   int _iGridRowCount, 
								   float32 _fWindowMinX, 
								   float32 _fWindowMinY, 
								   float32 _fWindowMaxX, 
								   float32 _fWindowMaxY)
{
	if (m_bInitialized)	{
		clear();
	}

	m_iGridColCount = _iGridColCount;
	m_iGridRowCount = _iGridRowCount;

	m_fWindowMinX = _fWindowMinX;
	m_fWindowMinY = _fWindowMinY;
	m_fWindowMaxX = _fWindowMaxX;
	m_fWindowMaxY = _fWindowMaxY;

	_calculateDependents();

	m_bInitialized = _check();
	return m_bInitialized;
}

void CVolumeGeometry2D::_calculateDependents()
{
	m_iGridTotCount = (m_iGridColCount * m_iGridRowCount);

	m_fWindowLengthX = (m_fWindowMaxX - m_fWindowMinX);
	m_fWindowLengthY = (m_fWindowMaxY - m_fWindowMinY);
	m_fWindowArea = (m_fWindowLengthX * m_fWindowLengthY);

	m_fPixelLengthX = (m_fWindowLengthX / (float32)m_iGridColCount);
	m_fPixelLengthY = (m_fWindowLengthY / (float32)m_iGridRowCount);
	m_fPixelArea = (m_fPixelLengthX * m_fPixelLengthY);

	m_fDivPixelLengthX = ((float32)m_iGridColCount / m_fWindowLengthX); // == (1.0f / m_fPixelLengthX);
	m_fDivPixelLengthY = ((float32)m_iGridRowCount / m_fWindowLengthY); // == (1.0f / m_fPixelLengthY);
}

//----------------------------------------------------------------------------------------
// is of type
bool CVolumeGeometry2D::isEqual(const CVolumeGeometry2D &_pGeom2) const
{
	// both objects must be initialized
	if (!m_bInitialized || !_pGeom2.m_bInitialized) return false;

	// check all values
	if (m_iGridColCount != _pGeom2.m_iGridColCount)		return false;
	if (m_iGridRowCount != _pGeom2.m_iGridRowCount)		return false;
	if (m_iGridTotCount != _pGeom2.m_iGridTotCount)		return false;
	if (m_fWindowLengthX != _pGeom2.m_fWindowLengthX)		return false;
	if (m_fWindowLengthY != _pGeom2.m_fWindowLengthY)		return false;
	if (m_fWindowArea != _pGeom2.m_fWindowArea)			return false;
	if (m_fPixelLengthX != _pGeom2.m_fPixelLengthX)		return false;
	if (m_fPixelLengthY != _pGeom2.m_fPixelLengthY)		return false;
	if (m_fPixelArea != _pGeom2.m_fPixelArea)				return false;
	if (m_fDivPixelLengthX != _pGeom2.m_fDivPixelLengthX)	return false;
	if (m_fDivPixelLengthY != _pGeom2.m_fDivPixelLengthY)	return false;
	if (m_fWindowMinX != _pGeom2.m_fWindowMinX)			return false;
	if (m_fWindowMinY != _pGeom2.m_fWindowMinY)			return false;
	if (m_fWindowMaxX != _pGeom2.m_fWindowMaxX)			return false;
	if (m_fWindowMaxY != _pGeom2.m_fWindowMaxY)			return false;
	
	return true;
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CVolumeGeometry2D::getConfiguration() const 
{
	ConfigWriter CW("VolumeGeometry2D");

	CW.addInt("GridColCount", m_iGridColCount);
	CW.addInt("GridRowCount", m_iGridRowCount);
	CW.addOptionNumerical("WindowMinX", m_fWindowMinX);
	CW.addOptionNumerical("WindowMaxX", m_fWindowMaxX);
	CW.addOptionNumerical("WindowMinY", m_fWindowMinY);
	CW.addOptionNumerical("WindowMaxY", m_fWindowMaxY);

	return CW.getConfig();
}
//----------------------------------------------------------------------------------------

} // namespace astra
