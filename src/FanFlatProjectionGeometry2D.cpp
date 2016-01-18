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

#include "astra/FanFlatProjectionGeometry2D.h"

#include <cstring>
#include <sstream>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor. Sets all variables to zero. 
CFanFlatProjectionGeometry2D::CFanFlatProjectionGeometry2D()
{
	_clear();
	m_fOriginSourceDistance = 0.0f;
	m_fOriginDetectorDistance = 0.0f;
}

//----------------------------------------------------------------------------------------
// Constructor.
CFanFlatProjectionGeometry2D::CFanFlatProjectionGeometry2D(int _iProjectionAngleCount, 
														   int _iDetectorCount, 
														   float32 _fDetectorWidth, 
														   const float32* _pfProjectionAngles,
														   float32 _fOriginSourceDistance, 
														   float32 _fOriginDetectorDistance)
{
	this->initialize(_iProjectionAngleCount, 
					 _iDetectorCount, 
					 _fDetectorWidth, 
					 _pfProjectionAngles, 
					 _fOriginSourceDistance, 
					 _fOriginDetectorDistance);
}

//----------------------------------------------------------------------------------------
// Copy Constructor
CFanFlatProjectionGeometry2D::CFanFlatProjectionGeometry2D(const CFanFlatProjectionGeometry2D& _projGeom)
{
	_clear();
	this->initialize(_projGeom.m_iProjectionAngleCount, 
					 _projGeom.m_iDetectorCount, 
					 _projGeom.m_fDetectorWidth, 
					 _projGeom.m_pfProjectionAngles, 
					 _projGeom.m_fOriginSourceDistance, 
					 _projGeom.m_fOriginDetectorDistance);
}

//----------------------------------------------------------------------------------------
// Assignment operator.
CFanFlatProjectionGeometry2D& CFanFlatProjectionGeometry2D::operator=(const CFanFlatProjectionGeometry2D& _other)
{
	if (m_bInitialized)
		delete[] m_pfProjectionAngles;
	m_bInitialized = _other.m_bInitialized;
	if (m_bInitialized) {
		m_iProjectionAngleCount = _other.m_iProjectionAngleCount;
		m_iDetectorCount = _other.m_iDetectorCount;
		m_fDetectorWidth = _other.m_fDetectorWidth;
		m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
		memcpy(m_pfProjectionAngles, _other.m_pfProjectionAngles, sizeof(float32)*m_iProjectionAngleCount);
		m_fOriginSourceDistance = _other.m_fOriginSourceDistance;
		m_fOriginDetectorDistance = _other.m_fOriginDetectorDistance;
	}
	return *this;
}
//----------------------------------------------------------------------------------------
// Destructor.
CFanFlatProjectionGeometry2D::~CFanFlatProjectionGeometry2D()
{

}


//----------------------------------------------------------------------------------------
// Initialization.
bool CFanFlatProjectionGeometry2D::initialize(int _iProjectionAngleCount, 
											  int _iDetectorCount, 
											  float32 _fDetectorWidth, 
											  const float32* _pfProjectionAngles,
											  float32 _fOriginSourceDistance, 
											  float32 _fOriginDetectorDistance)
{
	m_fOriginSourceDistance = _fOriginSourceDistance;
	m_fOriginDetectorDistance = _fOriginDetectorDistance;
	_initialize(_iProjectionAngleCount, 
			    _iDetectorCount, 
			    _fDetectorWidth, 
			    _pfProjectionAngles);

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization with a Config object
bool CFanFlatProjectionGeometry2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry2D> CC("FanFlatProjectionGeometry2D", this, _cfg);		

	// initialization of parent class
	CProjectionGeometry2D::initialize(_cfg);

	// Required: DistanceOriginDetector
	XMLNode node = _cfg.self.getSingleNode("DistanceOriginDetector");
	ASTRA_CONFIG_CHECK(node, "FanFlatProjectionGeometry2D", "No DistanceOriginDetector tag specified.");
	m_fOriginDetectorDistance = node.getContentNumerical();
	CC.markNodeParsed("DistanceOriginDetector");

	// Required: DetectorOriginSource
	node = _cfg.self.getSingleNode("DistanceOriginSource");
	ASTRA_CONFIG_CHECK(node, "FanFlatProjectionGeometry2D", "No DistanceOriginSource tag specified.");
	m_fOriginSourceDistance = node.getContentNumerical();
	CC.markNodeParsed("DistanceOriginSource");

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry2D* CFanFlatProjectionGeometry2D::clone()
{
	return new CFanFlatProjectionGeometry2D(*this);
}

//----------------------------------------------------------------------------------------
// is equal
bool CFanFlatProjectionGeometry2D::isEqual(CProjectionGeometry2D* _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CFanFlatProjectionGeometry2D
	CFanFlatProjectionGeometry2D* pGeom2 = dynamic_cast<CFanFlatProjectionGeometry2D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorCount != pGeom2->m_iDetectorCount) return false;
	if (m_fDetectorWidth != pGeom2->m_fDetectorWidth) return false;
	if (m_fOriginSourceDistance != pGeom2->m_fOriginSourceDistance) return false;
	if (m_fOriginDetectorDistance != pGeom2->m_fOriginDetectorDistance) return false;
	
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		if (m_pfProjectionAngles[i] != pGeom2->m_pfProjectionAngles[i]) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// Is of type
bool CFanFlatProjectionGeometry2D::isOfType(const std::string& _sType)
{
	 return (_sType == "fanflat");
}

CVector3D CFanFlatProjectionGeometry2D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex /* = 0 */)
{
	CVector3D vOutput(0.0f, 0.0f, 0.0f);

	// not implemented
	ASTRA_ASSERT(false);

	return vOutput;
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CFanFlatProjectionGeometry2D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("ProjectionGeometry2D");
	cfg->self.addAttribute("type", "fanflat");
	cfg->self.addChildNode("DetectorCount", getDetectorCount());
	cfg->self.addChildNode("DetectorWidth", getDetectorWidth());
	cfg->self.addChildNode("DistanceOriginSource", getOriginSourceDistance());
	cfg->self.addChildNode("DistanceOriginDetector", getOriginDetectorDistance());
	cfg->self.addChildNode("ProjectionAngles", m_pfProjectionAngles, m_iProjectionAngleCount);
	return cfg;
}
//----------------------------------------------------------------------------------------


} // namespace astra
