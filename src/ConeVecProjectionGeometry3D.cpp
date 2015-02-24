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

#include "astra/ConeVecProjectionGeometry3D.h"

#include <cstring>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CConeVecProjectionGeometry3D::CConeVecProjectionGeometry3D() :
	CProjectionGeometry3D() 
{
	m_pProjectionAngles = 0;
}

//----------------------------------------------------------------------------------------
// Constructor.
CConeVecProjectionGeometry3D::CConeVecProjectionGeometry3D(int _iProjectionAngleCount, 
                                                                   int _iDetectorRowCount, 
                                                                   int _iDetectorColCount, 
                                                                   const SConeProjection* _pProjectionAngles
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
CConeVecProjectionGeometry3D::~CConeVecProjectionGeometry3D()
{
	delete[] m_pProjectionAngles;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CConeVecProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ConeVecProjectionGeometry3D", this, _cfg);	

	XMLNode* node;

	// TODO: Fix up class hierarchy... this class doesn't fit very well.
	// initialization of parent class
	//CProjectionGeometry3D::initialize(_cfg);

	// Required: DetectorRowCount
	node = _cfg.self->getSingleNode("DetectorRowCount");
	ASTRA_CONFIG_CHECK(node, "ConeVecProjectionGeometry3D", "No DetectorRowCount tag specified.");
	m_iDetectorRowCount = boost::lexical_cast<int>(node->getContent());
	ASTRA_DELETE(node);
	CC.markNodeParsed("DetectorRowCount");

	// Required: DetectorColCount
	node = _cfg.self->getSingleNode("DetectorColCount");
	ASTRA_CONFIG_CHECK(node, "ConeVecProjectionGeometry3D", "No DetectorColCount tag specified.");
	m_iDetectorColCount = boost::lexical_cast<int>(node->getContent());
	m_iDetectorTotCount = m_iDetectorRowCount * m_iDetectorColCount;
	ASTRA_DELETE(node);
	CC.markNodeParsed("DetectorColCount");

	// Required: Vectors
	node = _cfg.self->getSingleNode("Vectors");
	ASTRA_CONFIG_CHECK(node, "ConeVecProjectionGeometry3D", "No Vectors tag specified.");
	vector<double> data = node->getContentNumericalArrayDouble();
	CC.markNodeParsed("Vectors");
	ASTRA_DELETE(node);
	ASTRA_CONFIG_CHECK(data.size() % 12 == 0, "ConeVecProjectionGeometry3D", "Vectors doesn't consist of 12-tuples.");
	m_iProjectionAngleCount = data.size() / 12;
	m_pProjectionAngles = new SConeProjection[m_iProjectionAngleCount];

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SConeProjection& p = m_pProjectionAngles[i];
		p.fSrcX  = data[12*i +  0];
		p.fSrcY  = data[12*i +  1];
		p.fSrcZ  = data[12*i +  2];
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
bool CConeVecProjectionGeometry3D::initialize(int _iProjectionAngleCount, 
                                                  int _iDetectorRowCount, 
                                                  int _iDetectorColCount, 
                                                  const SConeProjection* _pProjectionAngles)
{
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorRowCount = _iDetectorRowCount;
	m_iDetectorColCount = _iDetectorColCount;
	m_pProjectionAngles = new SConeProjection[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; ++i)
		m_pProjectionAngles[i] = _pProjectionAngles[i];

	// TODO: check?

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry3D* CConeVecProjectionGeometry3D::clone() const
{
	CConeVecProjectionGeometry3D* res = new CConeVecProjectionGeometry3D();
	res->m_bInitialized				= m_bInitialized;
	res->m_iProjectionAngleCount	= m_iProjectionAngleCount;
	res->m_iDetectorRowCount		= m_iDetectorRowCount;
	res->m_iDetectorColCount		= m_iDetectorColCount;
	res->m_iDetectorTotCount		= m_iDetectorTotCount;
	res->m_fDetectorSpacingX		= m_fDetectorSpacingX;
	res->m_fDetectorSpacingY		= m_fDetectorSpacingY;
	res->m_pProjectionAngles		= new SConeProjection[m_iProjectionAngleCount];
	memcpy(res->m_pProjectionAngles, m_pProjectionAngles, sizeof(m_pProjectionAngles[0])*m_iProjectionAngleCount);
	return res;
}

//----------------------------------------------------------------------------------------
// is equal
bool CConeVecProjectionGeometry3D::isEqual(const CProjectionGeometry3D * _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CConeProjectionGeometry3D
	const CConeVecProjectionGeometry3D* pGeom2 = dynamic_cast<const CConeVecProjectionGeometry3D*>(_pGeom2);
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
bool CConeVecProjectionGeometry3D::isOfType(const std::string& _sType) const
{
	 return (_sType == "cone_vec");
}

//----------------------------------------------------------------------------------------
void CConeVecProjectionGeometry3D::toXML(XMLNode* _sNode) const
{
	_sNode->addAttribute("type","cone_vec");
	_sNode->addChildNode("DetectorRowCount", m_iDetectorRowCount);
	_sNode->addChildNode("DetectorColCount", m_iDetectorColCount);
	// TODO:
	//_sNode->addChildNode("ProjectionAngles", m_pfProjectionAngles, m_iProjectionAngleCount);
}

CVector3D CConeVecProjectionGeometry3D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex) const
{
	const SConeProjection& p = m_pProjectionAngles[_iProjectionIndex];
	int u = _iDetectorIndex % m_iDetectorColCount;
	int v = _iDetectorIndex / m_iDetectorColCount;

	return CVector3D(p.fDetSX + (u+0.5)*p.fDetUX + (v+0.5)*p.fDetVX - p.fSrcX, p.fDetSY + (u+0.5)*p.fDetUY + (v+0.5)*p.fDetVY - p.fSrcY, p.fDetSZ + (u+0.5)*p.fDetUZ + (v+0.5)*p.fDetVZ - p.fSrcZ);
}

void CConeVecProjectionGeometry3D::projectPoint(float32 fX, float32 fY, float32 fZ,
                                                 int iAngleIndex,
                                                 float32 &fU, float32 &fV) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	double fUX, fUY, fUZ, fUC;
	double fVX, fVY, fVZ, fVC;
	double fDX, fDY, fDZ, fDC;

	computeBP_UV_Coeffs(m_pProjectionAngles[iAngleIndex],
	                    fUX, fUY, fUZ, fUC, fVX, fVY, fVZ, fVC, fDX, fDY, fDZ, fDC);

	// The -0.5f shifts from corner to center of detector pixels
	double fD = fDX*fX + fDY*fY + fDZ*fZ + fDC;
	fU = (fUX*fX + fUY*fY + fUZ*fZ + fUC) / fD - 0.5f;
	fV = (fVX*fX + fVY*fY + fVZ*fZ + fVC) / fD - 0.5f;
}


//----------------------------------------------------------------------------------------

bool CConeVecProjectionGeometry3D::_check()
{
	// TODO
	return true;
}

} // end namespace astra
