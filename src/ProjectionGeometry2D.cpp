/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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
											 const float32* _pfProjectionAngles) : configCheckData(0)
{
	_clear();
	_initialize(_iAngleCount, _iDetectorCount, _fDetectorWidth, _pfProjectionAngles);
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
	}
	m_pfProjectionAngles = NULL;
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

	// Required: DetectorCount
	XMLNode node = _cfg.self.getSingleNode("DetectorCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry2D", "No DetectorCount tag specified.");
	try {
		m_iDetectorCount = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry2D", "DetectorCount must be an integer.");
	}
	CC.markNodeParsed("DetectorCount");

	if (!initializeAngles(_cfg))
		return false;

	// some checks
	ASTRA_CONFIG_CHECK(m_iDetectorCount > 0, "ProjectionGeometry2D", "DetectorCount should be positive.");

	return true;
}

bool CProjectionGeometry2D::initializeAngles(const Config& _cfg)
{
	ConfigStackCheck<CProjectionGeometry2D> CC("ProjectionGeometry2D", this, _cfg);

	// Required: DetectorWidth
	XMLNode node = _cfg.self.getSingleNode("DetectorWidth");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry2D", "No DetectorWidth tag specified.");
	try {
		m_fDetectorWidth = node.getContentNumerical();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry2D", "DetectorWidth must be numerical.");
	}
	CC.markNodeParsed("DetectorWidth");


	// Required: ProjectionAngles
	node = _cfg.self.getSingleNode("ProjectionAngles");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry2D", "No ProjectionAngles tag specified.");
	vector<float32> angles;
	try {
		angles = node.getContentNumericalArray();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry2D", "ProjectionAngles must be a numerical vector.");
	}
	m_iProjectionAngleCount = angles.size();
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry2D", "Not enough ProjectionAngles specified.");
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = angles[i];
	}
	CC.markNodeParsed("ProjectionAngles");
	ASTRA_CONFIG_CHECK(m_pfProjectionAngles != NULL, "ProjectionGeometry2D", "ProjectionAngles not initialized");

	ASTRA_CONFIG_CHECK(m_fDetectorWidth > 0.0f, "ProjectionGeometry2D", "DetectorWidth should be positive.");
	return true;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CProjectionGeometry2D::_initialize(int _iProjectionAngleCount, 
									    int _iDetectorCount, 
									    float32 _fDetectorWidth, 
									    const float32* _pfProjectionAngles)
{
	if (m_bInitialized) {
		clear();
	}

	// copy parameters
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorCount = _iDetectorCount;
	m_fDetectorWidth = _fDetectorWidth;
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = _pfProjectionAngles[i];		
	}

	// Interface class, so don't set m_bInitialized to true
	return true;
}
//---------------------------------------------------------------------------------------

} // namespace astra
