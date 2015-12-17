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

#include "astra/ParallelProjectionGeometry2D.h"

#include <cstring>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CParallelProjectionGeometry2D::CParallelProjectionGeometry2D() :
	CProjectionGeometry2D() 
{

}

//----------------------------------------------------------------------------------------
// Constructor.
CParallelProjectionGeometry2D::CParallelProjectionGeometry2D(int _iProjectionAngleCount, 
															 int _iDetectorCount, 
															 float32 _fDetectorWidth, 
															 const float32* _pfProjectionAngles,
															 const float32* _pfExtraDetectorOffsets)
{
	_clear();
	initialize(_iProjectionAngleCount,
				_iDetectorCount, 
				_fDetectorWidth, 
				_pfProjectionAngles,
				_pfExtraDetectorOffsets);
}

//----------------------------------------------------------------------------------------
CParallelProjectionGeometry2D::CParallelProjectionGeometry2D(const CParallelProjectionGeometry2D& _projGeom)
{
	_clear();
	initialize(_projGeom.m_iProjectionAngleCount,
				_projGeom.m_iDetectorCount, 
				_projGeom.m_fDetectorWidth, 
				_projGeom.m_pfProjectionAngles,
				_projGeom.m_pfExtraDetectorOffset);
}

//----------------------------------------------------------------------------------------

CParallelProjectionGeometry2D& CParallelProjectionGeometry2D::operator=(const CParallelProjectionGeometry2D& _other)
{
	if (m_bInitialized)
		delete[] m_pfProjectionAngles;
	m_bInitialized = _other.m_bInitialized;
	if (_other.m_bInitialized) {
		m_iProjectionAngleCount = _other.m_iProjectionAngleCount;
		m_iDetectorCount = _other.m_iDetectorCount;
		m_fDetectorWidth = _other.m_fDetectorWidth;
		m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
		memcpy(m_pfProjectionAngles, _other.m_pfProjectionAngles, sizeof(float32)*m_iProjectionAngleCount);
	}
	return *this;
	
}

//----------------------------------------------------------------------------------------
// Destructor.
CParallelProjectionGeometry2D::~CParallelProjectionGeometry2D()
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CParallelProjectionGeometry2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry2D> CC("ParallelProjectionGeometry2D", this, _cfg);	


	// initialization of parent class
	CProjectionGeometry2D::initialize(_cfg);

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CParallelProjectionGeometry2D::initialize(int _iProjectionAngleCount, 
											   int _iDetectorCount, 
											   float32 _fDetectorWidth, 
											   const float32* _pfProjectionAngles,
											   const float32* _pfExtraDetectorOffsets)
{
	_initialize(_iProjectionAngleCount, 
			    _iDetectorCount, 
			    _fDetectorWidth, 
			    _pfProjectionAngles,
				_pfExtraDetectorOffsets);

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry2D* CParallelProjectionGeometry2D::clone()
{
	return new CParallelProjectionGeometry2D(*this);
}

//----------------------------------------------------------------------------------------
// is equal
bool CParallelProjectionGeometry2D::isEqual(CProjectionGeometry2D* _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CParallelProjectionGeometry2D
	CParallelProjectionGeometry2D* pGeom2 = dynamic_cast<CParallelProjectionGeometry2D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorCount != pGeom2->m_iDetectorCount) return false;
	if (m_fDetectorWidth != pGeom2->m_fDetectorWidth) return false;
	
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
	//	if (m_pfProjectionAngles[i] != pGeom2->m_pfProjectionAngles[i]) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// is of type
bool CParallelProjectionGeometry2D::isOfType(const std::string& _sType)
{
	 return (_sType == "parallel");
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CParallelProjectionGeometry2D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("ProjectionGeometry2D");
	cfg->self.addAttribute("type", "parallel");
	cfg->self.addChildNode("DetectorCount", getDetectorCount());
	cfg->self.addChildNode("DetectorWidth", getDetectorWidth());
	cfg->self.addChildNode("ProjectionAngles", m_pfProjectionAngles, m_iProjectionAngleCount);
	if(m_pfExtraDetectorOffset!=NULL){
		XMLNode opt = cfg->self.addChildNode("Option");
		opt.addAttribute("key","ExtraDetectorOffset");
		opt.setContent(m_pfExtraDetectorOffset, m_iProjectionAngleCount);
	}
	return cfg;
}
//----------------------------------------------------------------------------------------

CVector3D CParallelProjectionGeometry2D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex /* = 0 */)
{
	CVector3D vOutput;

	float32 fProjectionAngle = getProjectionAngle(_iProjectionIndex);

	vOutput.setX(cosf(fProjectionAngle));
	vOutput.setY(sinf(fProjectionAngle));
	vOutput.setZ(0.0f);

	return vOutput;
}

} // end namespace astra
