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

#include "astra/VolumeGeometry3D.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// Check all variable values
bool CVolumeGeometry3D::_check()
{
	ASTRA_CONFIG_CHECK(m_iGridColCount > 0, "VolumeGeometry3D", "GridColCount must be strictly positive.");
	ASTRA_CONFIG_CHECK(m_iGridRowCount > 0, "VolumeGeometry3D", "GridRowCount must be strictly positive.");
	ASTRA_CONFIG_CHECK(m_iGridSliceCount > 0, "VolumeGeometry3D", "GridSliceCount must be strictly positive.");
	ASTRA_CONFIG_CHECK(m_fWindowMinX < m_fWindowMaxX, "VolumeGeometry3D", "WindowMinX should be lower than WindowMaxX.");
	ASTRA_CONFIG_CHECK(m_fWindowMinY < m_fWindowMaxY, "VolumeGeometry3D", "WindowMinY should be lower than WindowMaxY.");
	ASTRA_CONFIG_CHECK(m_fWindowMinZ < m_fWindowMaxZ, "VolumeGeometry3D", "WindowMinZ should be lower than WindowMaxZ.");

	ASTRA_CONFIG_CHECK(m_iGridTotCount == (m_iGridColCount * m_iGridRowCount * m_iGridSliceCount), "VolumeGeometry3D", "Internal configuration error.");
#if 0
	ASTRA_CONFIG_CHECK(m_fWindowLengthX == (m_fWindowMaxX - m_fWindowMinX), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fWindowLengthY == (m_fWindowMaxY - m_fWindowMinY), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fWindowLengthZ == (m_fWindowMaxZ - m_fWindowMinZ), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fWindowArea == (m_fWindowLengthX * m_fWindowLengthY *  m_fWindowLengthZ), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fPixelLengthX == (m_fWindowLengthX / (float32)m_iGridColCount), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fPixelLengthY == (m_fWindowLengthY / (float32)m_iGridRowCount), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fPixelLengthZ == (m_fWindowLengthZ / (float32)m_iGridSliceCount), "VolumeGeometry3D", "Internal configuration error.");

	ASTRA_CONFIG_CHECK(m_fPixelArea == (m_fPixelLengthX * m_fPixelLengthY * m_fPixelLengthZ), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fDivPixelLengthX == (1.0f / m_fPixelLengthX), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fDivPixelLengthY == (1.0f / m_fPixelLengthY), "VolumeGeometry3D", "Internal configuration error.");
	ASTRA_CONFIG_CHECK(m_fDivPixelLengthZ == (1.0f / m_fPixelLengthZ), "VolumeGeometry3D", "Internal configuration error.");
#endif

	return true;
}

//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
void CVolumeGeometry3D::clear()
{
	m_iGridColCount = 0;
	m_iGridRowCount = 0;
	m_iGridSliceCount = 0;
	m_iGridTotCount = 0;

	m_fWindowLengthX = 0.0f;
	m_fWindowLengthY = 0.0f;
	m_fWindowLengthZ = 0.0f;
	m_fWindowArea = 0.0f;

	m_fPixelLengthX = 0.0f;
	m_fPixelLengthY = 0.0f;
	m_fPixelLengthZ = 0.0f;
	m_fPixelArea = 0.0f;

	m_fDivPixelLengthX = 0.0f;  
	m_fDivPixelLengthY = 0.0f;
	m_fDivPixelLengthZ = 0.0f;

	m_fWindowMinX = 0.0f;
	m_fWindowMinY = 0.0f;
	m_fWindowMinZ = 0.0f;
	m_fWindowMaxX = 0.0f;
	m_fWindowMaxY = 0.0f;
	m_fWindowMaxZ = 0.0f;

	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Default constructor.
CVolumeGeometry3D::CVolumeGeometry3D() : configCheckData(0)
{
	clear();
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Default constructor
CVolumeGeometry3D::CVolumeGeometry3D(int _iGridColCount, int _iGridRowCount, int _iGridSliceCount)
 : configCheckData(0)
{
	clear();
	initialize(_iGridColCount, _iGridRowCount, _iGridSliceCount);
}

//----------------------------------------------------------------------------------------
// Constructor.
CVolumeGeometry3D::CVolumeGeometry3D(int _iGridColCount, 
									 int _iGridRowCount, 
									 int _iGridSliceCount, 
									 float32 _fWindowMinX, 
									 float32 _fWindowMinY,
									 float32 _fWindowMinZ,
									 float32 _fWindowMaxX,
									 float32 _fWindowMaxY,
									 float32 _fWindowMaxZ)
{
	clear();
	initialize(_iGridColCount, 
			   _iGridRowCount, 
			   _iGridSliceCount, 
			   _fWindowMinX, 
			   _fWindowMinY, 
			   _fWindowMinZ, 
			   _fWindowMaxX, 
			   _fWindowMaxY,
			   _fWindowMaxZ);
}

CVolumeGeometry3D::CVolumeGeometry3D(const CVolumeGeometry3D& _other)
{
	*this = _other;
}

CVolumeGeometry3D& CVolumeGeometry3D::operator=(const CVolumeGeometry3D& _other)
{
	m_bInitialized = _other.m_bInitialized;
	m_iGridColCount = _other.m_iGridColCount;
	m_iGridRowCount = _other.m_iGridRowCount;
	m_iGridSliceCount = _other.m_iGridSliceCount;
	m_fWindowLengthX = _other.m_fWindowLengthX;	
	m_fWindowLengthY = _other.m_fWindowLengthY;	
	m_fWindowLengthZ = _other.m_fWindowLengthZ;
	m_fWindowArea = _other.m_fWindowArea;
	m_fPixelLengthX = _other.m_fPixelLengthX;
	m_fPixelLengthY = _other.m_fPixelLengthY;
	m_fPixelLengthZ = _other.m_fPixelLengthZ;
	m_fDivPixelLengthX = _other.m_fDivPixelLengthX;
	m_fDivPixelLengthY = _other.m_fDivPixelLengthY;
	m_fDivPixelLengthZ = _other.m_fDivPixelLengthZ;
	m_fWindowMinX = _other.m_fWindowMinX;
	m_fWindowMinY = _other.m_fWindowMinY;
	m_fWindowMinZ = _other.m_fWindowMinZ;
	m_fWindowMaxX = _other.m_fWindowMaxX;
	m_fWindowMaxY = _other.m_fWindowMaxY;
	m_fWindowMaxZ = _other.m_fWindowMaxZ;

	m_iGridTotCount = _other.m_iGridTotCount;	
	m_fPixelArea = _other.m_fPixelArea;

	return *this;
}

//----------------------------------------------------------------------------------------
// Destructor.
CVolumeGeometry3D::~CVolumeGeometry3D()
{
	if (m_bInitialized) {
		clear();
	}
}

//----------------------------------------------------------------------------------------
// Initialization with a Config object
bool CVolumeGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CVolumeGeometry3D> CC("VolumeGeometry3D", this, _cfg);


	// uninitialize if the object was initialized before
	if (m_bInitialized)	{
		clear();
	}

	// Required: GridColCount
	XMLNode node = _cfg.self.getSingleNode("GridColCount");
	ASTRA_CONFIG_CHECK(node, "ReconstructionGeometry2D", "No GridColCount tag specified.");
	m_iGridColCount = node.getContentInt();
	CC.markNodeParsed("GridColCount");

	// Required: GridRowCount
	node = _cfg.self.getSingleNode("GridRowCount");
	ASTRA_CONFIG_CHECK(node, "ReconstructionGeometry2D", "No GridRowCount tag specified.");
	m_iGridRowCount = node.getContentInt();
	CC.markNodeParsed("GridRowCount");

	// Required: GridRowCount
	node = _cfg.self.getSingleNode("GridSliceCount");
	ASTRA_CONFIG_CHECK(node, "ReconstructionGeometry2D", "No GridSliceCount tag specified.");
	m_iGridSliceCount = node.getContentInt();
	CC.markNodeParsed("GridSliceCount");

	// Optional: Window minima and maxima
	m_fWindowMinX = _cfg.self.getOptionNumerical("WindowMinX", -m_iGridColCount/2.0f);
	m_fWindowMaxX = _cfg.self.getOptionNumerical("WindowMaxX", m_iGridColCount/2.0f);
	m_fWindowMinY = _cfg.self.getOptionNumerical("WindowMinY", -m_iGridRowCount/2.0f);
	m_fWindowMaxY = _cfg.self.getOptionNumerical("WindowMaxY", m_iGridRowCount/2.0f);
	m_fWindowMinZ = _cfg.self.getOptionNumerical("WindowMinZ", -m_iGridSliceCount/2.0f);
	m_fWindowMaxZ = _cfg.self.getOptionNumerical("WindowMaxZ", m_iGridSliceCount/2.0f);
	CC.markOptionParsed("WindowMinX");
	CC.markOptionParsed("WindowMaxX");
	CC.markOptionParsed("WindowMinY");
	CC.markOptionParsed("WindowMaxY");
	CC.markOptionParsed("WindowMinZ");
	CC.markOptionParsed("WindowMaxZ");

	// calculate some other things
	m_iGridTotCount = (m_iGridColCount * m_iGridRowCount * m_iGridSliceCount);
	m_fWindowLengthX = (m_fWindowMaxX - m_fWindowMinX);
	m_fWindowLengthY = (m_fWindowMaxY - m_fWindowMinY);
	m_fWindowLengthZ = (m_fWindowMaxZ - m_fWindowMinZ);
	m_fWindowArea = (m_fWindowLengthX * m_fWindowLengthY *  m_fWindowLengthZ);
	m_fPixelLengthX = (m_fWindowLengthX / (float32)m_iGridColCount);
	m_fPixelLengthY = (m_fWindowLengthY / (float32)m_iGridRowCount);
	m_fPixelLengthZ = (m_fWindowLengthZ / (float32)m_iGridSliceCount);

	m_fPixelArea = (m_fPixelLengthX * m_fPixelLengthY * m_fPixelLengthZ);
    m_fDivPixelLengthX = ((float32)m_iGridColCount / m_fWindowLengthX); // == (1.0f / m_fPixelLengthX);
	m_fDivPixelLengthY = ((float32)m_iGridRowCount / m_fWindowLengthY); // == (1.0f / m_fPixelLengthY);
	m_fDivPixelLengthZ = ((float32)m_iGridSliceCount / m_fWindowLengthZ); // == (1.0f / m_fPixelLengthZ);

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CVolumeGeometry3D::initialize(int _iGridColCount, int _iGridRowCount, int _iGridSliceCount)
{
	return initialize(_iGridColCount, 
					  _iGridRowCount, 
					  _iGridSliceCount, 
					  -_iGridColCount/2.0f,
					  -_iGridRowCount/2.0f,
					  -_iGridSliceCount/2.0f,
					  _iGridColCount/2.0f,
					  _iGridRowCount/2.0f,
					  _iGridSliceCount/2.0f);
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CVolumeGeometry3D::initialize(int _iGridColCount, 
								   int _iGridRowCount, 
								   int _iGridSliceCount,
								   float32 _fWindowMinX, 
								   float32 _fWindowMinY, 
								   float32 _fWindowMinZ, 
								   float32 _fWindowMaxX, 
								   float32 _fWindowMaxY,
								   float32 _fWindowMaxZ)
{
	if (m_bInitialized)	{
		clear();
	}

	m_iGridColCount = _iGridColCount;
	m_iGridRowCount = _iGridRowCount;
	m_iGridSliceCount = _iGridSliceCount;
	m_iGridTotCount = (m_iGridColCount * m_iGridRowCount * m_iGridSliceCount);

	m_fWindowMinX = _fWindowMinX;
	m_fWindowMinY = _fWindowMinY;
	m_fWindowMinZ = _fWindowMinZ;
	m_fWindowMaxX = _fWindowMaxX;
	m_fWindowMaxY = _fWindowMaxY;
	m_fWindowMaxZ = _fWindowMaxZ;

	m_fWindowLengthX = (m_fWindowMaxX - m_fWindowMinX);
	m_fWindowLengthY = (m_fWindowMaxY - m_fWindowMinY);
	m_fWindowLengthZ = (m_fWindowMaxZ - m_fWindowMinZ);
	m_fWindowArea = (m_fWindowLengthX * m_fWindowLengthY * m_fWindowLengthZ);

	m_fPixelLengthX = (m_fWindowLengthX / (float32)m_iGridColCount);
	m_fPixelLengthY = (m_fWindowLengthY / (float32)m_iGridRowCount);
	m_fPixelLengthZ = (m_fWindowLengthZ / (float32)m_iGridSliceCount);
	m_fPixelArea = (m_fPixelLengthX * m_fPixelLengthY);

    m_fDivPixelLengthX = ((float32)m_iGridColCount / m_fWindowLengthX); // == (1.0f / m_fPixelLengthX);
	m_fDivPixelLengthY = ((float32)m_iGridRowCount / m_fWindowLengthY); // == (1.0f / m_fPixelLengthY);
	m_fDivPixelLengthZ = ((float32)m_iGridSliceCount / m_fWindowLengthZ); // == (1.0f / m_fPixelLengthZ);

	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CVolumeGeometry3D* CVolumeGeometry3D::clone() const
{
	CVolumeGeometry3D* res = new CVolumeGeometry3D();
	res->m_bInitialized		= m_bInitialized;
	res->m_iGridColCount	= m_iGridColCount;
	res->m_iGridRowCount	= m_iGridRowCount;
	res->m_iGridSliceCount	= m_iGridSliceCount;
	res->m_iGridTotCount	= m_iGridTotCount;
	res->m_fWindowLengthX	= m_fWindowLengthX;
	res->m_fWindowLengthY	= m_fWindowLengthY;
	res->m_fWindowLengthZ	= m_fWindowLengthZ;
	res->m_fWindowArea		= m_fWindowArea;
	res->m_fPixelLengthX	= m_fPixelLengthX;
	res->m_fPixelLengthY	= m_fPixelLengthY;
	res->m_fPixelLengthZ	= m_fPixelLengthZ;
	res->m_fPixelArea		= m_fPixelArea;
    res->m_fDivPixelLengthX = m_fDivPixelLengthX;
	res->m_fDivPixelLengthY = m_fDivPixelLengthY;
	res->m_fDivPixelLengthZ = m_fDivPixelLengthZ;
	res->m_fWindowMinX		= m_fWindowMinX;
	res->m_fWindowMinY		= m_fWindowMinY;
	res->m_fWindowMinZ		= m_fWindowMinZ;
	res->m_fWindowMaxX		= m_fWindowMaxX;
	res->m_fWindowMaxY		= m_fWindowMaxY;
	res->m_fWindowMaxZ		= m_fWindowMaxZ;
	return res;
}

//----------------------------------------------------------------------------------------
// is of type
bool CVolumeGeometry3D::isEqual(const CVolumeGeometry3D* _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !_pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iGridColCount != _pGeom2->m_iGridColCount) return false;
	if (m_iGridRowCount != _pGeom2->m_iGridRowCount) return false;
	if (m_iGridSliceCount != _pGeom2->m_iGridSliceCount) return false;
	if (m_iGridTotCount != _pGeom2->m_iGridTotCount) return false;
	if (m_fWindowLengthX != _pGeom2->m_fWindowLengthX) return false;
	if (m_fWindowLengthY != _pGeom2->m_fWindowLengthY) return false;
	if (m_fWindowLengthZ != _pGeom2->m_fWindowLengthZ) return false;
	if (m_fWindowArea != _pGeom2->m_fWindowArea) return false;
	if (m_fPixelLengthX != _pGeom2->m_fPixelLengthX) return false;
	if (m_fPixelLengthY != _pGeom2->m_fPixelLengthY) return false;
	if (m_fPixelLengthZ != _pGeom2->m_fPixelLengthZ) return false;
	if (m_fPixelArea != _pGeom2->m_fPixelArea) return false;
    if (m_fDivPixelLengthX != _pGeom2->m_fDivPixelLengthX) return false;
	if (m_fDivPixelLengthY != _pGeom2->m_fDivPixelLengthY) return false;
	if (m_fDivPixelLengthZ != _pGeom2->m_fDivPixelLengthZ) return false;
	if (m_fWindowMinX != _pGeom2->m_fWindowMinX) return false;
	if (m_fWindowMinY != _pGeom2->m_fWindowMinY) return false;
	if (m_fWindowMinZ != _pGeom2->m_fWindowMinZ) return false;
	if (m_fWindowMaxX != _pGeom2->m_fWindowMaxX) return false;
	if (m_fWindowMaxY != _pGeom2->m_fWindowMaxY) return false;
	if (m_fWindowMaxZ != _pGeom2->m_fWindowMaxZ) return false;
	
	return true;
}

CVolumeGeometry2D * CVolumeGeometry3D::createVolumeGeometry2D() const
{
	CVolumeGeometry2D * pOutput = new CVolumeGeometry2D();
	pOutput->initialize(getGridColCount(), getGridRowCount());
	return pOutput;
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CVolumeGeometry3D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("VolumeGeometry3D");

	cfg->self.addChildNode("GridColCount", m_iGridColCount);
	cfg->self.addChildNode("GridRowCount", m_iGridRowCount);
	cfg->self.addChildNode("GridSliceCount", m_iGridSliceCount);

	cfg->self.addOption("WindowMinX", m_fWindowMinX);
	cfg->self.addOption("WindowMaxX", m_fWindowMaxX);
	cfg->self.addOption("WindowMinY", m_fWindowMinY);
	cfg->self.addOption("WindowMaxY", m_fWindowMaxY);
	cfg->self.addOption("WindowMinZ", m_fWindowMinZ);
	cfg->self.addOption("WindowMaxZ", m_fWindowMaxZ);

	return cfg;
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------

} // namespace astra
