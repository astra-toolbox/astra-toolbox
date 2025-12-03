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

#include "astra/ProjectionGeometry2D.h"
#include "astra/Logging.h"

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CProjectionGeometry2D::CProjectionGeometry2D()
	: m_bInitialized(false),
	  m_iProjectionAngleCount(0),
	  m_iDetectorCount(0),
	  m_fDetectorWidth(0.0f),
	  configCheckData(nullptr)
{

}

//----------------------------------------------------------------------------------------
// Constructor.
CProjectionGeometry2D::CProjectionGeometry2D(int _iAngleCount, 
                                             int _iDetectorCount,
                                             float32 _fDetectorWidth,
                                             std::vector<float32> &&_pfProjectionAngles)
	: CProjectionGeometry2D()
{
	_initialize(_iAngleCount, _iDetectorCount, _fDetectorWidth,
	            std::move(_pfProjectionAngles));
}

//----------------------------------------------------------------------------------------
// Check all variable values.
bool CProjectionGeometry2D::_check()
{
	ASTRA_CONFIG_CHECK(m_iDetectorCount > 0, "ProjectionGeometry2D", "Detector Count should be positive.");
	ASTRA_CONFIG_CHECK(m_fDetectorWidth > 0.0f, "ProjectionGeometry2D", "Detector Width should be positive.");
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry2D", "ProjectionAngleCount should be positive.");
	ASTRA_CONFIG_CHECK(m_pfProjectionAngles.size() == m_iProjectionAngleCount, "ProjectionGeometry2D", "Number of angles does not match");

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
	assert(!m_bInitialized);

	ConfigReader<CProjectionGeometry2D> CR("ProjectionGeometry2D", this, _cfg);

	bool ok = true;

	// Required: number of voxels
	ok &= CR.getRequiredInt("DetectorCount", m_iDetectorCount);

	if (!ok)
		return false;

	if (!initializeAngles(_cfg))
		return false;

	// some checks
	ASTRA_CONFIG_CHECK(m_iDetectorCount > 0, "ProjectionGeometry2D", "DetectorCount should be positive.");

	return true;
}

bool CProjectionGeometry2D::initializeAngles(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry2D> CR("ProjectionGeometry2D", this, _cfg);

	bool ok = true;

	// Required: Detector pixel dimensions
	ok &= CR.getRequiredNumerical("DetectorWidth", m_fDetectorWidth);

	if (!ok)
		return false;

	// Required: ProjectionAngles
	vector<double> angles;
	if (!CR.getRequiredNumericalArray("ProjectionAngles", angles))
		return false;
	m_iProjectionAngleCount = angles.size();
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry2D", "Not enough ProjectionAngles specified.");
	m_pfProjectionAngles.resize(m_iProjectionAngleCount);
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = (float)angles[i];
	}

	ASTRA_CONFIG_CHECK(m_fDetectorWidth > 0.0f, "ProjectionGeometry2D", "DetectorWidth should be positive.");
	return true;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CProjectionGeometry2D::_initialize(int _iProjectionAngleCount, 
                                        int _iDetectorCount,
                                        float32 _fDetectorWidth,
                                        std::vector<float32> &&_pfProjectionAngles)
{
	assert(!m_bInitialized);

	// copy parameters
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorCount = _iDetectorCount;
	m_fDetectorWidth = _fDetectorWidth;
	m_pfProjectionAngles = std::move(_pfProjectionAngles);

	return true;
}
//---------------------------------------------------------------------------------------

} // namespace astra
