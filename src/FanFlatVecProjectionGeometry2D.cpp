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

#include "astra/FanFlatVecProjectionGeometry2D.h"

#include <cstring>
#include <sstream>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor. Sets all variables to zero. 
CFanFlatVecProjectionGeometry2D::CFanFlatVecProjectionGeometry2D()
{
	_clear();
	m_pProjectionAngles = 0;
}

//----------------------------------------------------------------------------------------
// Constructor.
CFanFlatVecProjectionGeometry2D::CFanFlatVecProjectionGeometry2D(int _iProjectionAngleCount, 
                                                                 int _iDetectorCount, 
                                                                 const SFanProjection* _pProjectionAngles)
{
	this->initialize(_iProjectionAngleCount, 
	                 _iDetectorCount, 
	                 _pProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Copy Constructor
CFanFlatVecProjectionGeometry2D::CFanFlatVecProjectionGeometry2D(const CFanFlatVecProjectionGeometry2D& _projGeom)
{
	_clear();
	this->initialize(_projGeom.m_iProjectionAngleCount,
	                 _projGeom.m_iDetectorCount,
	                 _projGeom.m_pProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Assignment operator.
CFanFlatVecProjectionGeometry2D& CFanFlatVecProjectionGeometry2D::operator=(const CFanFlatVecProjectionGeometry2D& _other)
{
	if (m_bInitialized)
		delete[] m_pProjectionAngles;
	m_bInitialized = _other.m_bInitialized;
	if (m_bInitialized) {
		m_iProjectionAngleCount = _other.m_iProjectionAngleCount;
		m_iDetectorCount = _other.m_iDetectorCount;
		m_pProjectionAngles = new SFanProjection[m_iProjectionAngleCount];
		memcpy(m_pProjectionAngles, _other.m_pProjectionAngles, sizeof(m_pProjectionAngles[0])*m_iProjectionAngleCount);
	}
	return *this;
}
//----------------------------------------------------------------------------------------
// Destructor.
CFanFlatVecProjectionGeometry2D::~CFanFlatVecProjectionGeometry2D()
{
	// TODO
	delete[] m_pProjectionAngles;
}


//----------------------------------------------------------------------------------------
// Initialization.
bool CFanFlatVecProjectionGeometry2D::initialize(int _iProjectionAngleCount, 
											  int _iDetectorCount, 
											  const SFanProjection* _pProjectionAngles)
{
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorCount = _iDetectorCount;
	m_pProjectionAngles = new SFanProjection[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; ++i)
		m_pProjectionAngles[i] = _pProjectionAngles[i];

	// TODO: check?

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization with a Config object
bool CFanFlatVecProjectionGeometry2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry2D> CC("FanFlatVecProjectionGeometry2D", this, _cfg);	

	XMLNode node;

	// TODO: Fix up class hierarchy... this class doesn't fit very well.
	// initialization of parent class
	//CProjectionGeometry2D::initialize(_cfg);

	// Required: DetectorCount
	node = _cfg.self.getSingleNode("DetectorCount");
	ASTRA_CONFIG_CHECK(node, "FanFlatVecProjectionGeometry3D", "No DetectorRowCount tag specified.");
	m_iDetectorCount = node.getContentInt();
	CC.markNodeParsed("DetectorCount");

	// Required: Vectors
	node = _cfg.self.getSingleNode("Vectors");
	ASTRA_CONFIG_CHECK(node, "FanFlatVecProjectionGeometry3D", "No Vectors tag specified.");
	vector<float32> data = node.getContentNumericalArray();
	CC.markNodeParsed("Vectors");
	ASTRA_CONFIG_CHECK(data.size() % 6 == 0, "FanFlatVecProjectionGeometry3D", "Vectors doesn't consist of 6-tuples.");
	m_iProjectionAngleCount = data.size() / 6;
	m_pProjectionAngles = new SFanProjection[m_iProjectionAngleCount];

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SFanProjection& p = m_pProjectionAngles[i];
		p.fSrcX  = data[6*i +  0];
		p.fSrcY  = data[6*i +  1];
		p.fDetUX = data[6*i +  4];
		p.fDetUY = data[6*i +  5];

		// The backend code currently expects the corner of the detector, while
		// the matlab interface supplies the center
		p.fDetSX = data[6*i +  2] - 0.5f * m_iDetectorCount * p.fDetUX;
		p.fDetSY = data[6*i +  3] - 0.5f * m_iDetectorCount * p.fDetUY;
	}



	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry2D* CFanFlatVecProjectionGeometry2D::clone()
{
	return new CFanFlatVecProjectionGeometry2D(*this);
}

//----------------------------------------------------------------------------------------
// is equal
bool CFanFlatVecProjectionGeometry2D::isEqual(CProjectionGeometry2D* _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CFanFlatVecProjectionGeometry2D
	CFanFlatVecProjectionGeometry2D* pGeom2 = dynamic_cast<CFanFlatVecProjectionGeometry2D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorCount != pGeom2->m_iDetectorCount) return false;
	
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		if (memcmp(&m_pProjectionAngles[i], &pGeom2->m_pProjectionAngles[i], sizeof(m_pProjectionAngles[i])) != 0) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// Is of type
bool CFanFlatVecProjectionGeometry2D::isOfType(const std::string& _sType)
{
	return (_sType == "fanflat_vec");
}

//----------------------------------------------------------------------------------------

CVector3D CFanFlatVecProjectionGeometry2D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex /* = 0 */)
{
	CVector3D vOutput(0.0f, 0.0f, 0.0f);

	// not implemented
	ASTRA_ASSERT(false);

	return vOutput;
}

//----------------------------------------------------------------------------------------

void CFanFlatVecProjectionGeometry2D::getRayParams(int _iRow, int _iColumn, float32& _fT, float32& _fTheta) const
{
	// not implemented
	ASTRA_ASSERT(false);
}

//----------------------------------------------------------------------------------------

bool CFanFlatVecProjectionGeometry2D::_check()
{
	// TODO
	return true;
}


//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CFanFlatVecProjectionGeometry2D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("ProjectionGeometry2D");
	cfg->self.addAttribute("type", "fanflat_vec");
	cfg->self.addChildNode("DetectorCount", getDetectorCount());
	std::string vectors = "";
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SFanProjection& p = m_pProjectionAngles[i];
		vectors += StringUtil::toString(p.fSrcX) + ",";
		vectors += StringUtil::toString(p.fSrcY) + ",";
		vectors += StringUtil::toString(p.fDetSX + 0.5f * m_iDetectorCount * p.fDetUX) + ",";
		vectors += StringUtil::toString(p.fDetSY + 0.5f * m_iDetectorCount * p.fDetUY) + ",";
		vectors += StringUtil::toString(p.fDetUX) + ",";
		vectors += StringUtil::toString(p.fDetUY);
		if (i < m_iProjectionAngleCount-1) vectors += ';';
	}
	cfg->self.addChildNode("Vectors", vectors);
	return cfg;
}
//----------------------------------------------------------------------------------------


} // namespace astra
