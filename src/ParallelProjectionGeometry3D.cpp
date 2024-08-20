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

#include "astra/ParallelProjectionGeometry3D.h"

#include "astra/GeometryUtil3D.h"

#include <cstring>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CParallelProjectionGeometry3D::CParallelProjectionGeometry3D() :
	CProjectionGeometry3D() 
{

}

//----------------------------------------------------------------------------------------
// Constructor.
CParallelProjectionGeometry3D::CParallelProjectionGeometry3D(int _iProjectionAngleCount, 
					 									     int _iDetectorRowCount, 
															 int _iDetectorColCount, 
															 float32 _fDetectorWidth, 
															 float32 _fDetectorHeight, 
															 const float32* _pfProjectionAngles) :
	CProjectionGeometry3D() 
{
	initialize(_iProjectionAngleCount, 
			   _iDetectorRowCount, 
			   _iDetectorColCount, 
			   _fDetectorWidth, 
			   _fDetectorHeight, 
			   _pfProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Destructor.
CParallelProjectionGeometry3D::~CParallelProjectionGeometry3D()
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CParallelProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ParallelProjectionGeometry3D", this, _cfg);	
	

	// initialization of parent class
	if (!CProjectionGeometry3D::initialize(_cfg))
		return false;

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CParallelProjectionGeometry3D::initialize(int _iProjectionAngleCount, 
											   int _iDetectorRowCount, 
											   int _iDetectorColCount, 
											   float32 _fDetectorWidth, 
											   float32 _fDetectorHeight, 
											   const float32* _pfProjectionAngles)
{
	_initialize(_iProjectionAngleCount, 
			    _iDetectorRowCount, 
			    _iDetectorColCount, 
			    _fDetectorWidth, 
			    _fDetectorHeight, 
			    _pfProjectionAngles);

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry3D* CParallelProjectionGeometry3D::clone() const
{
	CParallelProjectionGeometry3D* res = new CParallelProjectionGeometry3D();
	res->m_bInitialized				= m_bInitialized;
	res->m_iProjectionAngleCount	= m_iProjectionAngleCount;
	res->m_iDetectorRowCount		= m_iDetectorRowCount;
	res->m_iDetectorColCount		= m_iDetectorColCount;
	res->m_iDetectorTotCount		= m_iDetectorTotCount;
	res->m_fDetectorSpacingX		= m_fDetectorSpacingX;
	res->m_fDetectorSpacingY		= m_fDetectorSpacingY;
	res->m_pfProjectionAngles		= new float32[m_iProjectionAngleCount];
	memcpy(res->m_pfProjectionAngles, m_pfProjectionAngles, sizeof(float32)*m_iProjectionAngleCount);
	return res;
}

//----------------------------------------------------------------------------------------
// is equal
bool CParallelProjectionGeometry3D::isEqual(const CProjectionGeometry3D * _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CParallelProjectionGeometry3D
	const CParallelProjectionGeometry3D* pGeom2 = dynamic_cast<const CParallelProjectionGeometry3D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorRowCount != pGeom2->m_iDetectorRowCount) return false;
	if (m_iDetectorColCount != pGeom2->m_iDetectorColCount) return false;
	if (m_iDetectorTotCount != pGeom2->m_iDetectorTotCount) return false;
	if (m_fDetectorSpacingX != pGeom2->m_fDetectorSpacingX) return false;
	if (m_fDetectorSpacingY != pGeom2->m_fDetectorSpacingY) return false;
	
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		if (m_pfProjectionAngles[i] != pGeom2->m_pfProjectionAngles[i]) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// is of type
bool CParallelProjectionGeometry3D::isOfType(const std::string& _sType) const
{
	 return (_sType == "parallel3d");
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CParallelProjectionGeometry3D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("ProjectionGeometry3D");
	cfg->self.addAttribute("type", "parallel3d");
	cfg->self.addChildNode("DetectorRowCount", m_iDetectorRowCount);
	cfg->self.addChildNode("DetectorColCount", m_iDetectorColCount);
	cfg->self.addChildNode("DetectorSpacingX", m_fDetectorSpacingX);
	cfg->self.addChildNode("DetectorSpacingY", m_fDetectorSpacingY);
	cfg->self.addChildNode("ProjectionAngles", m_pfProjectionAngles, m_iProjectionAngleCount);
	return cfg;
}
//----------------------------------------------------------------------------------------

void CParallelProjectionGeometry3D::projectPoint(double fX, double fY, double fZ,
                                                 int iAngleIndex,
                                                 double &fU, double &fV) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	// V (detector row)
	fV = detectorOffsetYToRowIndexFloat(fZ);

	// U (detector column)
	float alpha = m_pfProjectionAngles[iAngleIndex];
	// projector direction is (cos(alpha), sin(alpha))
	fU = detectorOffsetXToColIndexFloat(cos(alpha) * fX + sin(alpha) * fY);
}

CParallelProjectionGeometry2D * CParallelProjectionGeometry3D::createProjectionGeometry2D() const
{
	const float32 * pfProjectionAngles = getProjectionAngles(); //new float32[getProjectionCount()];
	//getProjectionAngles(pfProjectionAngles);
	
	CParallelProjectionGeometry2D * pOutput = new CParallelProjectionGeometry2D(getProjectionCount(), 
		getDetectorColCount(), getDetectorSpacingX(), pfProjectionAngles);

	//delete [] pfProjectionAngles;

	return pOutput;
}

//----------------------------------------------------------------------------------------

} // end namespace astra
