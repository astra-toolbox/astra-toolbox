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

#include "astra/ConeVecProjectionGeometry3D.h"

#include "astra/XMLConfig.h"
#include "astra/Utilities.h"
#include "astra/Logging.h"

#include <cstring>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CConeVecProjectionGeometry3D::CConeVecProjectionGeometry3D() :
	CProjectionGeometry3D() 
{

}

//----------------------------------------------------------------------------------------
// Constructor.
CConeVecProjectionGeometry3D::CConeVecProjectionGeometry3D(int _iProjectionAngleCount,
                                                           int _iDetectorRowCount,
                                                           int _iDetectorColCount,
                                                           std::vector<SConeProjection> &&_pProjectionAngles)
: CProjectionGeometry3D()
{
	initialize(_iProjectionAngleCount,
	           _iDetectorRowCount,
	           _iDetectorColCount,
	           std::move(_pProjectionAngles));
}

//----------------------------------------------------------------------------------------
// Destructor.
CConeVecProjectionGeometry3D::~CConeVecProjectionGeometry3D()
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CConeVecProjectionGeometry3D::initialize(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry3D> CR("ConeVecProjectionGeometry3D", this, _cfg);	

	// initialization of parent class
	if (!CProjectionGeometry3D::initialize(_cfg))
		return false;

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

bool CConeVecProjectionGeometry3D::initializeAngles(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry3D> CR("ConeVecProjectionGeometry3D", this, _cfg);

	// Required: Vectors
	vector<double> data;
	if (!CR.getRequiredNumericalArray("Vectors", data))
		return false;
	ASTRA_CONFIG_CHECK(data.size() % 12 == 0, "ConeVecProjectionGeometry3D", "Vectors doesn't consist of 12-tuples.");
	m_iProjectionAngleCount = data.size() / 12;
	m_ProjectionAngles.resize(m_iProjectionAngleCount);

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SConeProjection& p = m_ProjectionAngles[i];
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
		p.fDetSX = data[12*i +  3] - 0.5 * m_iDetectorRowCount * p.fDetVX - 0.5 * m_iDetectorColCount * p.fDetUX;
		p.fDetSY = data[12*i +  4] - 0.5 * m_iDetectorRowCount * p.fDetVY - 0.5 * m_iDetectorColCount * p.fDetUY;
		p.fDetSZ = data[12*i +  5] - 0.5 * m_iDetectorRowCount * p.fDetVZ - 0.5 * m_iDetectorColCount * p.fDetUZ;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CConeVecProjectionGeometry3D::initialize(int _iProjectionAngleCount, 
                                                  int _iDetectorRowCount, 
                                                  int _iDetectorColCount, 
                                                  std::vector<SConeProjection> &&_pProjectionAngles)
{
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorRowCount = _iDetectorRowCount;
	m_iDetectorColCount = _iDetectorColCount;
	m_ProjectionAngles = std::move(_pProjectionAngles);

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
	res->m_ProjectionAngles		= m_ProjectionAngles;
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
		if (memcmp(&m_ProjectionAngles[i], &pGeom2->m_ProjectionAngles[i], sizeof(m_ProjectionAngles[i])) != 0) return false;
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
// Get the configuration object
Config* CConeVecProjectionGeometry3D::getConfiguration() const 
{
	ConfigWriter CW("ProjectionGeometry3D", "cone_vec");

	CW.addInt("DetectorRowCount", m_iDetectorRowCount);
	CW.addInt("DetectorColCount", m_iDetectorColCount);

	std::vector<double> vectors;
	vectors.resize(12 * m_iProjectionAngleCount);

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		const SConeProjection& p = m_ProjectionAngles[i];

		vectors[12*i +  0] = p.fSrcX;
		vectors[12*i +  1] = p.fSrcY;
		vectors[12*i +  2] = p.fSrcZ;
		vectors[12*i +  3] = p.fDetSX + 0.5*m_iDetectorRowCount*p.fDetVX + 0.5*m_iDetectorColCount*p.fDetUX;
		vectors[12*i +  4] = p.fDetSY + 0.5*m_iDetectorRowCount*p.fDetVY + 0.5*m_iDetectorColCount*p.fDetUY;
		vectors[12*i +  5] = p.fDetSZ + 0.5*m_iDetectorRowCount*p.fDetVZ + 0.5*m_iDetectorColCount*p.fDetUZ;
		vectors[12*i +  6] = p.fDetUX;
		vectors[12*i +  7] = p.fDetUY;
		vectors[12*i +  8] = p.fDetUZ;
		vectors[12*i +  9] = p.fDetVX;
		vectors[12*i + 10] = p.fDetVY;
		vectors[12*i + 11] = p.fDetVZ;
	}
	CW.addNumericalMatrix("Vectors", &vectors[0], m_iProjectionAngleCount, 12);

	return CW.getConfig();
}
//----------------------------------------------------------------------------------------

void CConeVecProjectionGeometry3D::projectPoint(double fX, double fY, double fZ,
                                                 int iAngleIndex,
                                                 double &fU, double &fV) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	double fUX, fUY, fUZ, fUC;
	double fVX, fVY, fVZ, fVC;
	double fDX, fDY, fDZ, fDC;

	computeBP_UV_Coeffs(m_ProjectionAngles[iAngleIndex],
	                    fUX, fUY, fUZ, fUC, fVX, fVY, fVZ, fVC, fDX, fDY, fDZ, fDC);

	// The -0.5f shifts from corner to center of detector pixels
	double fD = fDX*fX + fDY*fY + fDZ*fZ + fDC;
	fU = (fUX*fX + fUY*fY + fUZ*fZ + fUC) / fD - 0.5;
	fV = (fVX*fX + fVY*fY + fVZ*fZ + fVC) / fD - 0.5;
}


//----------------------------------------------------------------------------------------

bool CConeVecProjectionGeometry3D::_check()
{
	ASTRA_CONFIG_CHECK(m_ProjectionAngles.size() == m_iProjectionAngleCount, "ConeVecProjectionGeometry3D", "Number of vectors does not match number of angles");

	// TODO
	return true;
}

} // end namespace astra
