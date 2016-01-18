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

#include "astra/ProjectionGeometry2D.h"

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CProjectionGeometry2D::CProjectionGeometry2D() : configCheckData(0)
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Constructor.
CProjectionGeometry2D::CProjectionGeometry2D(int _iAngleCount, 
											 int _iDetectorCount, 
											 float32 _fDetectorWidth, 
											 const float32* _pfProjectionAngles,
											 const float32* _pfExtraDetectorOffsets) : configCheckData(0)
{
	_clear();
	_initialize(_iAngleCount, _iDetectorCount, _fDetectorWidth, _pfProjectionAngles,_pfExtraDetectorOffsets);
}

//----------------------------------------------------------------------------------------
// Destructor.
CProjectionGeometry2D::~CProjectionGeometry2D()
{
	if (m_bInitialized)	{
		clear();
	}
}

//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
// Should only be used by constructors.  Otherwise use the clear() function.
void CProjectionGeometry2D::_clear()
{
	m_iProjectionAngleCount = 0;
	m_iDetectorCount = 0;
	m_fDetectorWidth = 0.0f;
	m_pfProjectionAngles = NULL;
	m_pfExtraDetectorOffset = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
void CProjectionGeometry2D::clear()
{
	m_iProjectionAngleCount = 0;
	m_iDetectorCount = 0;
	m_fDetectorWidth = 0.0f;
	if (m_bInitialized){
		delete[] m_pfProjectionAngles;
		delete[] m_pfExtraDetectorOffset;
	}
	m_pfProjectionAngles = NULL;
	m_pfExtraDetectorOffset = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check all variable values.
bool CProjectionGeometry2D::_check()
{
	ASTRA_CONFIG_CHECK(m_iDetectorCount > 0, "ProjectionGeometry2D", "Detector Count should be positive.");
	ASTRA_CONFIG_CHECK(m_fDetectorWidth > 0.0f, "ProjectionGeometry2D", "Detector Width should be positive.");
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry2D", "ProjectionAngleCount should be positive.");
	ASTRA_CONFIG_CHECK(m_pfProjectionAngles != NULL, "ProjectionGeometry2D", "ProjectionAngles not initialized");

	// autofix: angles in [0,2pi[
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		while (2*PI <= m_pfProjectionAngles[i]) m_pfProjectionAngles[i] -= 2*PI;
		while (m_pfProjectionAngles[i] < 0) m_pfProjectionAngles[i] += 2*PI;
	}

	// success
	return true;
}

//----------------------------------------------------------------------------------------
// Initialization with a Config object
bool CProjectionGeometry2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry2D> CC("ProjectionGeometry2D", this, _cfg);	

	// uninitialize if the object was initialized before
	if (m_bInitialized)	{
		clear();
	}

	// Required: DetectorWidth
	XMLNode node = _cfg.self.getSingleNode("DetectorWidth");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry2D", "No DetectorWidth tag specified.");
	m_fDetectorWidth = node.getContentNumerical();
	CC.markNodeParsed("DetectorWidth");

	// Required: DetectorCount
	node = _cfg.self.getSingleNode("DetectorCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry2D", "No DetectorCount tag specified.");
	m_iDetectorCount = node.getContentInt();
	CC.markNodeParsed("DetectorCount");

	// Required: ProjectionAngles
	node = _cfg.self.getSingleNode("ProjectionAngles");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry2D", "No ProjectionAngles tag specified.");
	vector<float32> angles = node.getContentNumericalArray();
	m_iProjectionAngleCount = angles.size();
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry2D", "Not enough ProjectionAngles specified.");
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = angles[i];
	}
	CC.markNodeParsed("ProjectionAngles");

	vector<float32> offset = _cfg.self.getOptionNumericalArray("ExtraDetectorOffset");
	m_pfExtraDetectorOffset = new float32[m_iProjectionAngleCount];
	if (offset.size() == (size_t)m_iProjectionAngleCount) {
		for (int i = 0; i < m_iProjectionAngleCount; i++) {
			m_pfExtraDetectorOffset[i] = offset[i];
		}
	} else {
		for (int i = 0; i < m_iProjectionAngleCount; i++) {
			m_pfExtraDetectorOffset[i] = 0.0f;
		}	
	}
	CC.markOptionParsed("ExtraDetectorOffset");

	// some checks
	ASTRA_CONFIG_CHECK(m_iDetectorCount > 0, "ProjectionGeometry2D", "DetectorCount should be positive.");
	ASTRA_CONFIG_CHECK(m_fDetectorWidth > 0.0f, "ProjectionGeometry2D", "DetectorWidth should be positive.");
	ASTRA_CONFIG_CHECK(m_pfProjectionAngles != NULL, "ProjectionGeometry2D", "ProjectionAngles not initialized");

	// Interface class, so don't return true
	return false;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CProjectionGeometry2D::_initialize(int _iProjectionAngleCount, 
									    int _iDetectorCount, 
									    float32 _fDetectorWidth, 
									    const float32* _pfProjectionAngles,
										const float32* _pfExtraDetectorOffsets)
{
	if (m_bInitialized) {
		clear();
	}

	// copy parameters
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorCount = _iDetectorCount;
	m_fDetectorWidth = _fDetectorWidth;
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	m_pfExtraDetectorOffset = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = _pfProjectionAngles[i];		
		m_pfExtraDetectorOffset[i] = _pfExtraDetectorOffsets ? _pfExtraDetectorOffsets[i]:0;
	}

	// Interface class, so don't set m_bInitialized to true
	return true;
}
//---------------------------------------------------------------------------------------

} // namespace astra
