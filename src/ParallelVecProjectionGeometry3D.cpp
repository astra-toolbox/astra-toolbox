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

#include "astra/ParallelVecProjectionGeometry3D.h"

#include <cstring>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CParallelVecProjectionGeometry3D::CParallelVecProjectionGeometry3D() :
	CProjectionGeometry3D() 
{
	m_pProjectionAngles = 0;
}

//----------------------------------------------------------------------------------------
// Constructor.
CParallelVecProjectionGeometry3D::CParallelVecProjectionGeometry3D(int _iProjectionAngleCount, 
                                                                   int _iDetectorRowCount, 
                                                                   int _iDetectorColCount, 
                                                                   const SPar3DProjection* _pProjectionAngles
															 ) :
	CProjectionGeometry3D() 
{
	initialize(_iProjectionAngleCount, 
	           _iDetectorRowCount, 
	           _iDetectorColCount, 
	           _pProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Destructor.
CParallelVecProjectionGeometry3D::~CParallelVecProjectionGeometry3D()
{
	delete[] m_pProjectionAngles;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CParallelVecProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ParallelVecProjectionGeometry3D", this, _cfg);	

	XMLNode node;

	// TODO: Fix up class hierarchy... this class doesn't fit very well.
	// initialization of parent class
	//CProjectionGeometry3D::initialize(_cfg);

	// Required: DetectorRowCount
	node = _cfg.self.getSingleNode("DetectorRowCount");
	ASTRA_CONFIG_CHECK(node, "ParallelVecProjectionGeometry3D", "No DetectorRowCount tag specified.");
	m_iDetectorRowCount = boost::lexical_cast<int>(node.getContent());
	CC.markNodeParsed("DetectorRowCount");

	// Required: DetectorCount
	node = _cfg.self.getSingleNode("DetectorColCount");
	ASTRA_CONFIG_CHECK(node, "", "No DetectorColCount tag specified.");
	m_iDetectorColCount = boost::lexical_cast<int>(node.getContent());
	m_iDetectorTotCount = m_iDetectorRowCount * m_iDetectorColCount;
	CC.markNodeParsed("DetectorColCount");

	// Required: Vectors
	node = _cfg.self.getSingleNode("Vectors");
	ASTRA_CONFIG_CHECK(node, "ParallelVecProjectionGeometry3D", "No Vectors tag specified.");
	vector<double> data = node.getContentNumericalArrayDouble();
	CC.markNodeParsed("Vectors");
	ASTRA_CONFIG_CHECK(data.size() % 12 == 0, "ParallelVecProjectionGeometry3D", "Vectors doesn't consist of 12-tuples.");
	m_iProjectionAngleCount = data.size() / 12;
	m_pProjectionAngles = new SPar3DProjection[m_iProjectionAngleCount];

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SPar3DProjection& p = m_pProjectionAngles[i];
		p.fRayX  = data[12*i +  0];
		p.fRayY  = data[12*i +  1];
		p.fRayZ  = data[12*i +  2];
		p.fDetUX = data[12*i +  6];
		p.fDetUY = data[12*i +  7];
		p.fDetUZ = data[12*i +  8];
		p.fDetVX = data[12*i +  9];
		p.fDetVY = data[12*i + 10];
		p.fDetVZ = data[12*i + 11];

		// The backend code currently expects the corner of the detector, while
		// the matlab interface supplies the center
		p.fDetSX = data[12*i +  3] - 0.5f * m_iDetectorRowCount * p.fDetVX - 0.5f * m_iDetectorColCount * p.fDetUX;
		p.fDetSY = data[12*i +  4] - 0.5f * m_iDetectorRowCount * p.fDetVY - 0.5f * m_iDetectorColCount * p.fDetUY;
		p.fDetSZ = data[12*i +  5] - 0.5f * m_iDetectorRowCount * p.fDetVZ - 0.5f * m_iDetectorColCount * p.fDetUZ;
	}

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CParallelVecProjectionGeometry3D::initialize(int _iProjectionAngleCount, 
                                                  int _iDetectorRowCount, 
                                                  int _iDetectorColCount, 
                                                  const SPar3DProjection* _pProjectionAngles)
{
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorRowCount = _iDetectorRowCount;
	m_iDetectorColCount = _iDetectorColCount;
	m_pProjectionAngles = new SPar3DProjection[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; ++i)
		m_pProjectionAngles[i] = _pProjectionAngles[i];

	// TODO: check?

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry3D* CParallelVecProjectionGeometry3D::clone() const
{
	CParallelVecProjectionGeometry3D* res = new CParallelVecProjectionGeometry3D();
	res->m_bInitialized				= m_bInitialized;
	res->m_iProjectionAngleCount	= m_iProjectionAngleCount;
	res->m_iDetectorRowCount		= m_iDetectorRowCount;
	res->m_iDetectorColCount		= m_iDetectorColCount;
	res->m_iDetectorTotCount		= m_iDetectorTotCount;
	res->m_fDetectorSpacingX		= m_fDetectorSpacingX;
	res->m_fDetectorSpacingY		= m_fDetectorSpacingY;
	res->m_pProjectionAngles		= new SPar3DProjection[m_iProjectionAngleCount];
	memcpy(res->m_pProjectionAngles, m_pProjectionAngles, sizeof(m_pProjectionAngles[0])*m_iProjectionAngleCount);
	return res;
}

//----------------------------------------------------------------------------------------
// is equal
bool CParallelVecProjectionGeometry3D::isEqual(const CProjectionGeometry3D * _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CParallelProjectionGeometry3D
	const CParallelVecProjectionGeometry3D* pGeom2 = dynamic_cast<const CParallelVecProjectionGeometry3D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorRowCount != pGeom2->m_iDetectorRowCount) return false;
	if (m_iDetectorColCount != pGeom2->m_iDetectorColCount) return false;
	if (m_iDetectorTotCount != pGeom2->m_iDetectorTotCount) return false;
	//if (m_fDetectorSpacingX != pGeom2->m_fDetectorSpacingX) return false;
	//if (m_fDetectorSpacingY != pGeom2->m_fDetectorSpacingY) return false;
	
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		if (memcmp(&m_pProjectionAngles[i], &pGeom2->m_pProjectionAngles[i], sizeof(m_pProjectionAngles[i])) != 0) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// is of type
bool CParallelVecProjectionGeometry3D::isOfType(const std::string& _sType) const
{
	 return (_sType == "parallel3d_vec");
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CParallelVecProjectionGeometry3D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("ProjectionGeometry3D");

	cfg->self.addAttribute("type", "parallel3d_vec");
	cfg->self.addChildNode("DetectorRowCount", m_iDetectorRowCount);
	cfg->self.addChildNode("DetectorColCount", m_iDetectorColCount);

	std::string vectors = "";
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SPar3DProjection& p = m_pProjectionAngles[i];
		vectors += boost::lexical_cast<string>(p.fRayX) + ",";
		vectors += boost::lexical_cast<string>(p.fRayY) + ",";
		vectors += boost::lexical_cast<string>(p.fRayZ) + ",";
		vectors += boost::lexical_cast<string>(p.fDetSX + 0.5f*m_iDetectorRowCount*p.fDetVX + 0.5f*m_iDetectorColCount*p.fDetUX) + ",";
		vectors += boost::lexical_cast<string>(p.fDetSY + 0.5f*m_iDetectorRowCount*p.fDetVY + 0.5f*m_iDetectorColCount*p.fDetUY) + ",";
		vectors += boost::lexical_cast<string>(p.fDetSZ + 0.5f*m_iDetectorRowCount*p.fDetVZ + 0.5f*m_iDetectorColCount*p.fDetUZ) + ",";
		vectors += boost::lexical_cast<string>(p.fDetUX) + ",";
		vectors += boost::lexical_cast<string>(p.fDetUY) + ",";
		vectors += boost::lexical_cast<string>(p.fDetUZ) + ",";
		vectors += boost::lexical_cast<string>(p.fDetVX) + ",";
		vectors += boost::lexical_cast<string>(p.fDetVY) + ",";
		vectors += boost::lexical_cast<string>(p.fDetVZ);
		if (i < m_iProjectionAngleCount-1) vectors += ';';
	}
	cfg->self.addChildNode("Vectors", vectors);

	return cfg;
}
//----------------------------------------------------------------------------------------

CVector3D CParallelVecProjectionGeometry3D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex) const
{
	const SPar3DProjection& p = m_pProjectionAngles[_iProjectionIndex];

	return CVector3D(p.fRayX, p.fRayY, p.fRayZ);
}

void CParallelVecProjectionGeometry3D::projectPoint(double fX, double fY, double fZ,
                                                    int iAngleIndex,
                                                    double &fU, double &fV) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	double fUX, fUY, fUZ, fUC;
	double fVX, fVY, fVZ, fVC;

	computeBP_UV_Coeffs(m_pProjectionAngles[iAngleIndex],
	                    fUX, fUY, fUZ, fUC, fVX, fVY, fVZ, fVC);

	// The -0.5f shifts from corner to center of detector pixels
	fU = (fUX*fX + fUY*fY + fUZ*fZ + fUC) - 0.5f;
	fV = (fVX*fX + fVY*fY + fVZ*fZ + fVC) - 0.5f;

}

void CParallelVecProjectionGeometry3D::backprojectPointX(int iAngleIndex, double fU, double fV,
	                               double fX, double &fY, double &fZ) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	SPar3DProjection &proj = m_pProjectionAngles[iAngleIndex];

	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fX - px) / proj.fRayX;

	fY = py + a * proj.fRayY;
	fZ = pz + a * proj.fRayZ;
}

void CParallelVecProjectionGeometry3D::backprojectPointY(int iAngleIndex, double fU, double fV,
	                               double fY, double &fX, double &fZ) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	SPar3DProjection &proj = m_pProjectionAngles[iAngleIndex];

	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fY - py) / proj.fRayY;

	fX = px + a * proj.fRayX;
	fZ = pz + a * proj.fRayZ;
}

void CParallelVecProjectionGeometry3D::backprojectPointZ(int iAngleIndex, double fU, double fV,
	                               double fZ, double &fX, double &fY) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	SPar3DProjection &proj = m_pProjectionAngles[iAngleIndex];

	double px = proj.fDetSX + fU * proj.fDetUX + fV * proj.fDetVX;
	double py = proj.fDetSY + fU * proj.fDetUY + fV * proj.fDetVY;
	double pz = proj.fDetSZ + fU * proj.fDetUZ + fV * proj.fDetVZ;

	double a = (fZ - pz) / proj.fRayZ;

	fX = px + a * proj.fRayX;
	fY = py + a * proj.fRayY;
}


//----------------------------------------------------------------------------------------

bool CParallelVecProjectionGeometry3D::_check()
{
	// TODO
	return true;
}

} // end namespace astra
