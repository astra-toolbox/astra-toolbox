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

#include "astra/ProjectionGeometry3D.h"
#include "astra/Logging.h"

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
// Initialization with a Config object
bool CProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ProjectionGeometry3D", this, _cfg);

	if (m_bInitialized) {
		clear();
	}

	ASTRA_ASSERT(_cfg.self);
	

	// Required: DetectorRowCount
	XMLNode node = _cfg.self.getSingleNode("DetectorRowCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorRowCount tag specified.");
	try {
		m_iDetectorRowCount = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry3D", "DetectorRowCount must be an integer.");
	}
	CC.markNodeParsed("DetectorRowCount");

	// Required: DetectorCount
	node = _cfg.self.getSingleNode("DetectorColCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorColCount tag specified.");
	try {
		m_iDetectorColCount = node.getContentInt();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry3D", "DetectorColCount must be an integer.");
	}
	m_iDetectorTotCount = m_iDetectorRowCount * m_iDetectorColCount;
	CC.markNodeParsed("DetectorColCount");


	if (!initializeAngles(_cfg))
		return false;

	return true;
}

bool CProjectionGeometry3D::initializeAngles(const Config& _cfg)
{
	ConfigStackCheck<CProjectionGeometry3D> CC("ProjectionGeometry3D", this, _cfg);

	// Required: DetectorWidth
	XMLNode node = _cfg.self.getSingleNode("DetectorSpacingX");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorSpacingX tag specified.");
	try {
		m_fDetectorSpacingX = node.getContentNumerical();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry3D", "DetectorSpacingX must be numerical.");
	}
	CC.markNodeParsed("DetectorSpacingX");

	// Required: DetectorHeight
	node = _cfg.self.getSingleNode("DetectorSpacingY");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorSpacingY tag specified.");
	try {
		m_fDetectorSpacingY = node.getContentNumerical();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry3D", "DetectorSpacingY must be numerical.");
	}
	CC.markNodeParsed("DetectorSpacingY");

	// Required: ProjectionAngles
	node = _cfg.self.getSingleNode("ProjectionAngles");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No ProjectionAngles tag specified.");
	vector<float32> angles;
	try {
		angles = node.getContentNumericalArray();
	} catch (const StringUtil::bad_cast &e) {
		ASTRA_CONFIG_CHECK(false, "ProjectionGeometry3D", "ProjectionAngles must be a numerical vector.");
	}
	m_iProjectionAngleCount = angles.size();
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry3D", "Not enough ProjectionAngles specified.");
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = angles[i];
	}
	CC.markNodeParsed("ProjectionAngles");


	return true;
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

	return true;
}

void CProjectionGeometry3D::getProjectedBBox(double fXMin, double fXMax,
                                             double fYMin, double fYMax,
                                             double fZMin, double fZMax,
                                             double &fUMin, double &fUMax,
                                             double &fVMin, double &fVMax) const
{
	double vmin_g, vmax_g;
	double umin_g, umax_g;

	// Default implementation, correct for flat panel detectors:
	// Project corners of volume, take bounding box

	assert(getProjectionCount() > 0);
	for (int i = 0; i < getProjectionCount(); ++i) {

		double vol_u[8];
		double vol_v[8];

		projectPoint(fXMin, fYMin, fZMin, i, vol_u[0], vol_v[0]);
		projectPoint(fXMin, fYMin, fZMax, i, vol_u[1], vol_v[1]);
		projectPoint(fXMin, fYMax, fZMin, i, vol_u[2], vol_v[2]);
		projectPoint(fXMin, fYMax, fZMax, i, vol_u[3], vol_v[3]);
		projectPoint(fXMax, fYMin, fZMin, i, vol_u[4], vol_v[4]);
		projectPoint(fXMax, fYMin, fZMax, i, vol_u[5], vol_v[5]);
		projectPoint(fXMax, fYMax, fZMin, i, vol_u[6], vol_v[6]);
		projectPoint(fXMax, fYMax, fZMax, i, vol_u[7], vol_v[7]);

		double umin = vol_u[0];
		double umax = vol_u[0];
		double vmin = vol_v[0];
		double vmax = vol_v[0];

		for (int j = 1; j < 8; ++j) {
			if (vol_u[j] < umin)
				umin = vol_u[j];
			if (vol_u[j] > umax)
				umax = vol_u[j];
			if (vol_v[j] < vmin)
				vmin = vol_v[j];
			if (vol_v[j] > vmax)
				vmax = vol_v[j];
		}

		if (i == 0 || umin < umin_g)
			umin_g = umin;
		if (i == 0 || umax > umax_g)
			umax_g = umax;
		if (i == 0 || vmin < vmin_g)
			vmin_g = vmin;
		if (i == 0 || vmax > vmax_g)
			vmax_g = vmax;
	}

	fUMin = umin_g;
	fUMax = umax_g;
	fVMin = vmin_g;
	fVMax = vmax_g;
}


} // namespace astra
