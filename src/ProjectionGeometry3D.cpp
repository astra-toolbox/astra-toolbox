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

#include "astra/ProjectionGeometry3D.h"

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Check all variable values.
bool CProjectionGeometry3D::_check()
{
	ASTRA_CONFIG_CHECK(m_iDetectorRowCount > 0, "ProjectionGeometry3D", "DetectorRowCount should be positive.");
	ASTRA_CONFIG_CHECK(m_iDetectorColCount > 0, "ProjectionGeometry3D", "DetectorColCount should be positive.");
	ASTRA_CONFIG_CHECK(m_fDetectorSpacingX > 0.0f, "ProjectionGeometry3D", "m_fDetectorSpacingX should be positive.");
	ASTRA_CONFIG_CHECK(m_fDetectorSpacingY > 0.0f, "ProjectionGeometry3D", "m_fDetectorSpacingY should be positive.");
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry3D", "ProjectionAngleCount should be positive.");
	ASTRA_CONFIG_CHECK(m_pfProjectionAngles != NULL, "ProjectionGeometry3D", "ProjectionAngles not initialized");

/*
	// autofix: angles in [0,2pi[
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		while (2*PI <= m_pfProjectionAngles[i]) m_pfProjectionAngles[i] -= 2*PI;
		while (m_pfProjectionAngles[i] < 0) m_pfProjectionAngles[i] += 2*PI;
	}
*/

	// succes
	return true;
}

//----------------------------------------------------------------------------------------
// Default constructor.
CProjectionGeometry3D::CProjectionGeometry3D() : configCheckData(0)
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Constructor.
CProjectionGeometry3D::CProjectionGeometry3D(int _iAngleCount, 
											 int _iDetectorRowCount, 
											 int _iDetectorColCount, 
											 float32 _fDetectorSpacingX, 
											 float32 _fDetectorSpacingY, 
											 const float32 *_pfProjectionAngles) : configCheckData(0)
{
	_clear();
	_initialize(_iAngleCount, 
			    _iDetectorRowCount, 
			    _iDetectorColCount, 
			    _fDetectorSpacingX, 
			    _fDetectorSpacingY, 
			    _pfProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Copy constructor.
CProjectionGeometry3D::CProjectionGeometry3D(const CProjectionGeometry3D& _projGeom)
{
	_clear();
	_initialize(_projGeom.m_iProjectionAngleCount,
				_projGeom.m_iDetectorRowCount, 
				_projGeom.m_iDetectorColCount, 
				_projGeom.m_fDetectorSpacingX, 
				_projGeom.m_fDetectorSpacingY, 
				_projGeom.m_pfProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Destructor.
CProjectionGeometry3D::~CProjectionGeometry3D()
{
	if (m_bInitialized)	{
		clear();
	}
}

//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
// Should only be used by constructors.  Otherwise use the clear() function.
void CProjectionGeometry3D::_clear()
{
	m_iProjectionAngleCount = 0;
	m_iDetectorRowCount = 0;
	m_iDetectorColCount = 0;
	m_fDetectorSpacingX = 0.0f;
	m_fDetectorSpacingY = 0.0f;
	m_pfProjectionAngles = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
void CProjectionGeometry3D::clear()
{
	m_iProjectionAngleCount = 0;
	m_iDetectorRowCount = 0;
	m_iDetectorColCount = 0;
	m_fDetectorSpacingX = 0.0f;
	m_fDetectorSpacingY = 0.0f;
	if (m_pfProjectionAngles != NULL) {
		delete [] m_pfProjectionAngles;
	}
	m_pfProjectionAngles = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Initialization witha Config object
bool CProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ProjectionGeometry3D", this, _cfg);

	if (m_bInitialized) {
		clear();
	}

	ASTRA_ASSERT(_cfg.self);
	
	// Required: DetectorWidth
	XMLNode node = _cfg.self.getSingleNode("DetectorSpacingX");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorSpacingX tag specified.");
	m_fDetectorSpacingX = node.getContentNumerical();
	CC.markNodeParsed("DetectorSpacingX");

	// Required: DetectorHeight
	node = _cfg.self.getSingleNode("DetectorSpacingY");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorSpacingY tag specified.");
	m_fDetectorSpacingY = node.getContentNumerical();
	CC.markNodeParsed("DetectorSpacingY");

	// Required: DetectorRowCount
	node = _cfg.self.getSingleNode("DetectorRowCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorRowCount tag specified.");
	m_iDetectorRowCount = node.getContentInt();
	CC.markNodeParsed("DetectorRowCount");

	// Required: DetectorCount
	node = _cfg.self.getSingleNode("DetectorColCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorColCount tag specified.");
	m_iDetectorColCount = node.getContentInt();
	m_iDetectorTotCount = m_iDetectorRowCount * m_iDetectorColCount;
	CC.markNodeParsed("DetectorColCount");

	// Required: ProjectionAngles
	node = _cfg.self.getSingleNode("ProjectionAngles");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No ProjectionAngles tag specified.");
	vector<float32> angles = node.getContentNumericalArray();
	m_iProjectionAngleCount = angles.size();
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry3D", "Not enough ProjectionAngles specified.");
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = angles[i];
	}
	CC.markNodeParsed("ProjectionAngles");

	// Interface class, so don't return true
	return false;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CProjectionGeometry3D::_initialize(int _iProjectionAngleCount, 
										int _iDetectorRowCount, 
										int _iDetectorColCount, 
										float32 _fDetectorSpacingX, 
										float32 _fDetectorSpacingY, 
										const float32 *_pfProjectionAngles)
{
	if (m_bInitialized) {
		clear();
	}

	// copy parameters
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorRowCount = _iDetectorRowCount;
	m_iDetectorColCount = _iDetectorColCount;
	m_iDetectorTotCount = _iDetectorRowCount * _iDetectorColCount;
	m_fDetectorSpacingX = _fDetectorSpacingX;
	m_fDetectorSpacingY = _fDetectorSpacingY;
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = _pfProjectionAngles[i];
	}

	m_iDetectorTotCount = m_iProjectionAngleCount * m_iDetectorRowCount * m_iDetectorColCount;

	// Interface class, so don't return true
	return false;
}

} // namespace astra
