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

#include "astra/ParallelVecProjectionGeometry2D.h"

#include "astra/XMLConfig.h"
#include "astra/Logging.h"

#include <cstring>
#include <sstream>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor. Sets all variables to zero. 
CParallelVecProjectionGeometry2D::CParallelVecProjectionGeometry2D()
{
	_clear();
	m_pProjectionAngles = 0;
}

//----------------------------------------------------------------------------------------
// Constructor.
CParallelVecProjectionGeometry2D::CParallelVecProjectionGeometry2D(int _iProjectionAngleCount, 
                                                                 int _iDetectorCount, 
                                                                 const SParProjection* _pProjectionAngles)
{
	this->initialize(_iProjectionAngleCount, 
	                 _iDetectorCount, 
	                 _pProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Copy Constructor
CParallelVecProjectionGeometry2D::CParallelVecProjectionGeometry2D(const CParallelVecProjectionGeometry2D& _projGeom)
{
	_clear();
	this->initialize(_projGeom.m_iProjectionAngleCount,
	                 _projGeom.m_iDetectorCount,
	                 _projGeom.m_pProjectionAngles);
}

//----------------------------------------------------------------------------------------
// Destructor.
CParallelVecProjectionGeometry2D::~CParallelVecProjectionGeometry2D()
{
	// TODO
	delete[] m_pProjectionAngles;
}


//----------------------------------------------------------------------------------------
// Initialization.
bool CParallelVecProjectionGeometry2D::initialize(int _iProjectionAngleCount, 
											  int _iDetectorCount, 
											  const SParProjection* _pProjectionAngles)
{
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorCount = _iDetectorCount;
	m_pProjectionAngles = new SParProjection[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; ++i)
		m_pProjectionAngles[i] = _pProjectionAngles[i];

	// TODO: check?

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization with a Config object
bool CParallelVecProjectionGeometry2D::initialize(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry2D> CR("ParallelVecProjectionGeometry2D", this, _cfg);	

	// initialization of parent class
	if (!CProjectionGeometry2D::initialize(_cfg))
		return false;


	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

bool CParallelVecProjectionGeometry2D::initializeAngles(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry2D> CR("ParallelVecProjectionGeometry2D", this, _cfg);

	// Required: Vectors
	vector<double> data;
	if (!CR.getRequiredNumericalArray("Vectors", data))
		return false;
	ASTRA_CONFIG_CHECK(data.size() % 6 == 0, "ParallelVecProjectionGeometry2D", "Vectors doesn't consist of 6-tuples.");
	m_iProjectionAngleCount = data.size() / 6;
	m_pProjectionAngles = new SParProjection[m_iProjectionAngleCount];

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SParProjection& p = m_pProjectionAngles[i];
		p.fRayX  = data[6*i +  0];
		p.fRayY  = data[6*i +  1];
		p.fDetUX = data[6*i +  4];
		p.fDetUY = data[6*i +  5];

		// The backend code currently expects the corner of the detector, while
		// the matlab interface supplies the center
		p.fDetSX = data[6*i +  2] - 0.5 * m_iDetectorCount * p.fDetUX;
		p.fDetSY = data[6*i +  3] - 0.5 * m_iDetectorCount * p.fDetUY;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry2D* CParallelVecProjectionGeometry2D::clone() const
{
	return new CParallelVecProjectionGeometry2D(*this);
}

//----------------------------------------------------------------------------------------
// is equal
bool CParallelVecProjectionGeometry2D::isEqual(const CProjectionGeometry2D &_pGeom2) const
{
	// try to cast argument to CParallelVecProjectionGeometry2D
	const CParallelVecProjectionGeometry2D* pGeom2 = dynamic_cast<const CParallelVecProjectionGeometry2D*>(&_pGeom2);
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
bool CParallelVecProjectionGeometry2D::isOfType(const std::string& _sType)
{
	return (_sType == "parallel_vec");
}

//----------------------------------------------------------------------------------------

bool CParallelVecProjectionGeometry2D::_check()
{
	// TODO
	return true;
}


//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CParallelVecProjectionGeometry2D::getConfiguration() const 
{
	ConfigWriter CW("ProjectionGeometry2D", "parallel_vec");

	CW.addInt("DetectorCount", getDetectorCount());

	std::vector<double> vectors;
	vectors.resize(6 * m_iProjectionAngleCount);

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SParProjection& p = m_pProjectionAngles[i];
		vectors[6*i + 0] = p.fRayX;
		vectors[6*i + 1] = p.fRayY;
		vectors[6*i + 2] = p.fDetSX + 0.5 * m_iDetectorCount * p.fDetUX;
		vectors[6*i + 3] = p.fDetSY + 0.5 * m_iDetectorCount * p.fDetUY;
		vectors[6*i + 4] = p.fDetUX;
		vectors[6*i + 5] = p.fDetUY;
	}
	CW.addNumericalMatrix("Vectors", &vectors[0], m_iProjectionAngleCount, 6);

	return CW.getConfig();
}
//----------------------------------------------------------------------------------------


} // namespace astra
