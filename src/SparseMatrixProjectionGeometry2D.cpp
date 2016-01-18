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

#include "astra/SparseMatrixProjectionGeometry2D.h"

#include "astra/AstraObjectManager.h"


using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CSparseMatrixProjectionGeometry2D::CSparseMatrixProjectionGeometry2D() :
	CProjectionGeometry2D() 
{
	m_pMatrix = 0;
}

//----------------------------------------------------------------------------------------
// Constructor.
CSparseMatrixProjectionGeometry2D::CSparseMatrixProjectionGeometry2D(int _iProjectionAngleCount, 
															 int _iDetectorCount,
															 const CSparseMatrix* _pMatrix)
{
	_clear();
	initialize(_iProjectionAngleCount,
				_iDetectorCount, 
				_pMatrix);
}

//----------------------------------------------------------------------------------------
CSparseMatrixProjectionGeometry2D::CSparseMatrixProjectionGeometry2D(const CSparseMatrixProjectionGeometry2D& _projGeom)
{
	_clear();
	initialize(_projGeom.m_iProjectionAngleCount,
				_projGeom.m_iDetectorCount, 
				_projGeom.m_pMatrix);
}

//----------------------------------------------------------------------------------------

CSparseMatrixProjectionGeometry2D& CSparseMatrixProjectionGeometry2D::operator=(const CSparseMatrixProjectionGeometry2D& _other)
{
	m_bInitialized = _other.m_bInitialized;
	if (_other.m_bInitialized) {
		m_pMatrix = _other.m_pMatrix;
		m_iDetectorCount = _other.m_iDetectorCount;
		m_fDetectorWidth = _other.m_fDetectorWidth;
	}
	return *this;
	
}

//----------------------------------------------------------------------------------------
// Destructor.
CSparseMatrixProjectionGeometry2D::~CSparseMatrixProjectionGeometry2D()
{
	m_pMatrix = 0;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CSparseMatrixProjectionGeometry2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry2D> CC("SparseMatrixProjectionGeometry2D", this, _cfg);	

	// initialization of parent class
	CProjectionGeometry2D::initialize(_cfg);

	// get matrix
	XMLNode node = _cfg.self.getSingleNode("MatrixID");
	ASTRA_CONFIG_CHECK(node, "SparseMatrixProjectionGeometry2D", "No MatrixID tag specified.");
	int id = node.getContentInt();
	m_pMatrix = CMatrixManager::getSingleton().get(id);
	CC.markNodeParsed("MatrixID");

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CSparseMatrixProjectionGeometry2D::initialize(int _iProjectionAngleCount, 
											   int _iDetectorCount, 
											   const CSparseMatrix* _pMatrix)
{
	if (m_bInitialized) {
		clear();
	}

	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorCount = _iDetectorCount;

	// FIXME: We should probably require these for consistency?
	m_fDetectorWidth = 1.0f;
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; ++i)
		m_pfProjectionAngles[i] = 0.0f;

	m_pMatrix = _pMatrix;

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Check.
bool CSparseMatrixProjectionGeometry2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjectionGeometry2D::_check(), "SparseMatrixProjectionGeometry2D", "Error in ProjectionGeometry2D initialization");

	ASTRA_CONFIG_CHECK(m_pMatrix, "SparseMatrixProjectionGeometry2D", "No matrix specified");

	ASTRA_CONFIG_CHECK(m_pMatrix->m_iHeight == (unsigned int)(m_iProjectionAngleCount * m_iDetectorCount), "SparseMatrixProjectionGeometry2D", "Matrix height doesn't match projection geometry");

	return true;
}


//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry2D* CSparseMatrixProjectionGeometry2D::clone()
{
	return new CSparseMatrixProjectionGeometry2D(*this);
}

//----------------------------------------------------------------------------------------
// is equal
bool CSparseMatrixProjectionGeometry2D::isEqual(CProjectionGeometry2D* _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CSparseMatrixProjectionGeometry2D
	CSparseMatrixProjectionGeometry2D* pGeom2 = dynamic_cast<CSparseMatrixProjectionGeometry2D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorCount != pGeom2->m_iDetectorCount) return false;
	if (m_fDetectorWidth != pGeom2->m_fDetectorWidth) return false;

	// Maybe check equality of matrices by element?
	if (m_pMatrix != pGeom2->m_pMatrix) return false;	

	return true;
}

//----------------------------------------------------------------------------------------
// is of type
bool CSparseMatrixProjectionGeometry2D::isOfType(const std::string& _sType)
{
	 return (_sType == "sparse_matrix");
}
//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CSparseMatrixProjectionGeometry2D::getConfiguration() const 
{
	Config* cfg = new Config();
	cfg->initialize("ProjectionGeometry2D");
	cfg->self.addAttribute("type", "sparse matrix");
	cfg->self.addChildNode("DetectorCount", getDetectorCount());
	cfg->self.addChildNode("DetectorWidth", getDetectorWidth());
	cfg->self.addChildNode("ProjectionAngles", m_pfProjectionAngles, m_iProjectionAngleCount);
	cfg->self.addChildNode("MatrixID", CMatrixManager::getSingleton().getIndex(m_pMatrix));
	return cfg;
}

//----------------------------------------------------------------------------------------
CVector3D CSparseMatrixProjectionGeometry2D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex)
{
	CVector3D vOutput(0.0f, 0.0f, 0.0f);

	// not implemented, yet
	ASTRA_ASSERT(false);

	return vOutput;
}

} // end namespace astra
