/*
-----------------------------------------------------------------------
Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
                2014, CWI, Amsterdam

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

#include "astra/ProjectionGeometry3D.h"

#include <boost/lexical_cast.hpp>

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
											 const float32 *_pfProjectionAngles,
											 const float32 *_pfExtraDetectorOffsetsX,
											 const float32 *_pfExtraDetectorOffsetsY) : configCheckData(0)
{
	_clear();
	_initialize(_iAngleCount, 
			    _iDetectorRowCount, 
			    _iDetectorColCount, 
			    _fDetectorSpacingX, 
			    _fDetectorSpacingY, 
			    _pfProjectionAngles,
				_pfExtraDetectorOffsetsX,
				_pfExtraDetectorOffsetsY);
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
				_projGeom.m_pfProjectionAngles,
				_projGeom.m_pfExtraDetectorOffsetsX,
				_projGeom.m_pfExtraDetectorOffsetsY);
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
	m_pfExtraDetectorOffsetsX = NULL;
	m_pfExtraDetectorOffsetsY = NULL;
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
	if (m_pfExtraDetectorOffsetsX != NULL) {
		delete [] m_pfExtraDetectorOffsetsX;
	}
	if (m_pfExtraDetectorOffsetsY != NULL) {
		delete [] m_pfExtraDetectorOffsetsY;
	}
	m_pfProjectionAngles = NULL;
	m_pfExtraDetectorOffsetsX = NULL;
	m_pfExtraDetectorOffsetsY = NULL;
	m_bInitialized = false;
}

//----------------------------------------------------------------------------------------
// Initialization witha Config object
bool CProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ProjectionGeometry3D", this, _cfg);

	if (m_bInitialized) {
		clear();
	}

	ASTRA_ASSERT(_cfg.self);
	
	// Required: DetectorWidth
	XMLNode* node = _cfg.self->getSingleNode("DetectorSpacingX");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorSpacingX tag specified.");
	m_fDetectorSpacingX = boost::lexical_cast<float32>(node->getContent());
	ASTRA_DELETE(node);
	CC.markNodeParsed("DetectorSpacingX");

	// Required: DetectorHeight
	node = _cfg.self->getSingleNode("DetectorSpacingY");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorSpacingY tag specified.");
	m_fDetectorSpacingY = boost::lexical_cast<float32>(node->getContent());
	ASTRA_DELETE(node);
	CC.markNodeParsed("DetectorSpacingY");

	// Required: DetectorRowCount
	node = _cfg.self->getSingleNode("DetectorRowCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorRowCount tag specified.");
	m_iDetectorRowCount = boost::lexical_cast<int>(node->getContent());
	ASTRA_DELETE(node);
	CC.markNodeParsed("DetectorRowCount");

	// Required: DetectorCount
	node = _cfg.self->getSingleNode("DetectorColCount");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No DetectorColCount tag specified.");
	m_iDetectorColCount = boost::lexical_cast<int>(node->getContent());
	m_iDetectorTotCount = m_iDetectorRowCount * m_iDetectorColCount;
	ASTRA_DELETE(node);
	CC.markNodeParsed("DetectorColCount");

	// Required: ProjectionAngles
	node = _cfg.self->getSingleNode("ProjectionAngles");
	ASTRA_CONFIG_CHECK(node, "ProjectionGeometry3D", "No ProjectionAngles tag specified.");
	vector<float32> angles = node->getContentNumericalArray();
	m_iProjectionAngleCount = angles.size();
	ASTRA_CONFIG_CHECK(m_iProjectionAngleCount > 0, "ProjectionGeometry3D", "Not enough ProjectionAngles specified.");
	m_pfProjectionAngles = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = angles[i];
	}
	CC.markNodeParsed("ProjectionAngles");
	ASTRA_DELETE(node);

	// Optional: ExtraDetectorOffsetX
	node = _cfg.self->getSingleNode("ExtraDetectorOffsetsX");
	m_pfExtraDetectorOffsetsX = new float32[m_iProjectionAngleCount];
	if (node) {
		vector<float32> translationsX = node->getContentNumericalArray();
		if (translationsX.size() < m_iProjectionAngleCount){
			cout << "Not enough ExtraDetectorOffsetsX components specified. " << endl;
			for (int i = 0; i < m_iProjectionAngleCount; i++) {
				m_pfExtraDetectorOffsetsX[i] = 0;
			}
		}
		else {
			for (int i = 0; i < m_iProjectionAngleCount; i++) {
				m_pfExtraDetectorOffsetsX[i] = translationsX[i];
			}
		}
	}
	else {
		//cout << "No ExtraDetectorOffsetsX tag specified." << endl;
		for (int i = 0; i < m_iProjectionAngleCount; i++) {
			m_pfExtraDetectorOffsetsX[i] = 0;
		}
	}
	CC.markOptionParsed("ExtraDetectorOffsetsX");
	ASTRA_DELETE(node);

	// Optional: ExtraDetectorOffsetsY
	node = _cfg.self->getSingleNode("ExtraDetectorOffsetsY");
	m_pfExtraDetectorOffsetsY = new float32[m_iProjectionAngleCount];
	if (node) {
		vector<float32> translationsX = node->getContentNumericalArray();
		if (translationsX.size() < m_iProjectionAngleCount){
			cout << "Not enough ExtraDetectorOffsetsY components specified. " << endl;
			for (int i = 0; i < m_iProjectionAngleCount; i++) {
				m_pfExtraDetectorOffsetsY[i] = 0;
			}
		}
		else {
			for (int i = 0; i < m_iProjectionAngleCount; i++) {
				m_pfExtraDetectorOffsetsY[i] = translationsX[i];
			}
		}
	}
	else {
		//cout << "No ExtraDetectorOffsetsY tag specified." << endl;
		for (int i = 0; i < m_iProjectionAngleCount; i++) {
			m_pfExtraDetectorOffsetsY[i] = 0;
		}
	}	
	CC.markOptionParsed("ExtraDetectorOffsetsY");
	ASTRA_DELETE(node);

	// Interface class, so don't return true
	return false;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CProjectionGeometry3D::_initialize(int _iProjectionAngleCount, 
										int _iDetectorRowCount, 
										int _iDetectorColCount, 
										float32 _fDetectorSpacingX, 
										float32 _fDetectorSpacingY, 
										const float32 *_pfProjectionAngles,
										const float32 *_pfExtraDetectorOffsetsX,
										const float32 *_pfExtraDetectorOffsetsY)
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
	m_pfExtraDetectorOffsetsX = new float32[m_iProjectionAngleCount];
	m_pfExtraDetectorOffsetsY = new float32[m_iProjectionAngleCount];
	for (int i = 0; i < m_iProjectionAngleCount; i++) {
		m_pfProjectionAngles[i] = _pfProjectionAngles[i];
		m_pfExtraDetectorOffsetsX[i] = _pfExtraDetectorOffsetsX ? _pfExtraDetectorOffsetsX[i]:0;
		m_pfExtraDetectorOffsetsY[i] = _pfExtraDetectorOffsetsY ? _pfExtraDetectorOffsetsY[i]:0;
		//m_pfExtraDetectorOffsetsX[i] = 0;
		//m_pfExtraDetectorOffsetsY[i] = 0;
	}

	m_iDetectorTotCount = m_iProjectionAngleCount * m_iDetectorRowCount * m_iDetectorColCount;

	// Interface class, so don't return true
	return false;
}

//---------------------------------------------------------------------------------------
// 
AstraError CProjectionGeometry3D::setExtraDetectorOffsetsX(float32* _pfExtraDetectorOffsetsX)
{
	if (!m_bInitialized)		
		return ASTRA_ERROR_NOT_INITIALIZED;

	for (int iAngle = 0; iAngle<m_iProjectionAngleCount; iAngle++)
		m_pfExtraDetectorOffsetsX[iAngle] = _pfExtraDetectorOffsetsX[iAngle];

	return ASTRA_SUCCESS;		
}

//---------------------------------------------------------------------------------------
// 
AstraError CProjectionGeometry3D::setExtraDetectorOffsetsY(float32* _pfExtraDetectorOffsetsY)
{
	if (!m_bInitialized)		
		return ASTRA_ERROR_NOT_INITIALIZED;

	for (int iAngle = 0; iAngle<m_iProjectionAngleCount; iAngle++)
		m_pfExtraDetectorOffsetsY[iAngle] = _pfExtraDetectorOffsetsY[iAngle];

	return ASTRA_SUCCESS;
}
} // namespace astra
