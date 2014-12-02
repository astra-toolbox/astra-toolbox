/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

#include "astra/ConeProjectionGeometry3D.h"

#include <boost/lexical_cast.hpp>
#include <cstring>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CConeProjectionGeometry3D::CConeProjectionGeometry3D() :
	CProjectionGeometry3D() 
{
	m_fOriginSourceDistance = 0.0f;
	m_fOriginDetectorDistance = 0.0f;
}

//----------------------------------------------------------------------------------------
// Constructor.
CConeProjectionGeometry3D::CConeProjectionGeometry3D(int _iProjectionAngleCount, 
					 									     int _iDetectorRowCount, 
															 int _iDetectorColCount, 
															 float32 _fDetectorWidth, 
															 float32 _fDetectorHeight, 
															 const float32* _pfProjectionAngles,
															 float32 _fOriginSourceDistance, 
															 float32 _fOriginDetectorDistance) :
	CProjectionGeometry3D() 
{
	initialize(_iProjectionAngleCount, 
			   _iDetectorRowCount, 
			   _iDetectorColCount, 
			   _fDetectorWidth, 
			   _fDetectorHeight, 
			   _pfProjectionAngles,
			   _fOriginSourceDistance,
			   _fOriginDetectorDistance);
}

//----------------------------------------------------------------------------------------
// Destructor.
CConeProjectionGeometry3D::~CConeProjectionGeometry3D()
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CConeProjectionGeometry3D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CProjectionGeometry3D> CC("ConeProjectionGeometry3D", this, _cfg);	

	// initialization of parent class
	CProjectionGeometry3D::initialize(_cfg);

	// Required: DistanceOriginDetector
	XMLNode* node = _cfg.self->getSingleNode("DistanceOriginDetector");
	ASTRA_CONFIG_CHECK(node, "ConeProjectionGeometry3D", "No DistanceOriginDetector tag specified.");
	m_fOriginDetectorDistance = boost::lexical_cast<float32>(node->getContent());
	ASTRA_DELETE(node);
	CC.markNodeParsed("DistanceOriginDetector");

	// Required: DetectorOriginSource
	node = _cfg.self->getSingleNode("DistanceOriginSource");
	ASTRA_CONFIG_CHECK(node, "ConeProjectionGeometry3D", "No DistanceOriginSource tag specified.");
	m_fOriginSourceDistance = boost::lexical_cast<float32>(node->getContent());
	ASTRA_DELETE(node);
	CC.markNodeParsed("DistanceOriginSource");

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CConeProjectionGeometry3D::initialize(int _iProjectionAngleCount, 
											   int _iDetectorRowCount, 
											   int _iDetectorColCount, 
											   float32 _fDetectorWidth, 
											   float32 _fDetectorHeight, 
											   const float32* _pfProjectionAngles,
											   float32 _fOriginSourceDistance, 
											   float32 _fOriginDetectorDistance)
{
	_initialize(_iProjectionAngleCount, 
			    _iDetectorRowCount, 
			    _iDetectorColCount, 
			    _fDetectorWidth, 
			    _fDetectorHeight, 
			    _pfProjectionAngles);

	m_fOriginSourceDistance = _fOriginSourceDistance;
	m_fOriginDetectorDistance = _fOriginDetectorDistance;

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry3D* CConeProjectionGeometry3D::clone() const
{
	CConeProjectionGeometry3D* res = new CConeProjectionGeometry3D();
	res->m_bInitialized				= m_bInitialized;
	res->m_iProjectionAngleCount	= m_iProjectionAngleCount;
	res->m_iDetectorRowCount		= m_iDetectorRowCount;
	res->m_iDetectorColCount		= m_iDetectorColCount;
	res->m_iDetectorTotCount		= m_iDetectorTotCount;
	res->m_fDetectorSpacingX		= m_fDetectorSpacingX;
	res->m_fDetectorSpacingY		= m_fDetectorSpacingY;
	res->m_pfProjectionAngles		= new float32[m_iProjectionAngleCount];
	memcpy(res->m_pfProjectionAngles, m_pfProjectionAngles, sizeof(float32)*m_iProjectionAngleCount);
	res->m_fOriginSourceDistance	= m_fOriginSourceDistance;
	res->m_fOriginDetectorDistance	= m_fOriginDetectorDistance;
	return res;
}

//----------------------------------------------------------------------------------------
// is equal
bool CConeProjectionGeometry3D::isEqual(const CProjectionGeometry3D* _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CParallelProjectionGeometry3D
	const CConeProjectionGeometry3D* pGeom2 = dynamic_cast<const CConeProjectionGeometry3D*>(_pGeom2);
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
	if (m_fOriginSourceDistance != pGeom2->m_fOriginSourceDistance) return false;
	if (m_fOriginDetectorDistance != pGeom2->m_fOriginDetectorDistance) return false;

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		if (m_pfProjectionAngles[i] != pGeom2->m_pfProjectionAngles[i]) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// is of type
bool CConeProjectionGeometry3D::isOfType(const std::string& _sType) const
{
	 return (_sType == "cone");
}

//----------------------------------------------------------------------------------------
void CConeProjectionGeometry3D::toXML(XMLNode* _sNode) const
{
	_sNode->addAttribute("type", "cone");
	_sNode->addChildNode("DetectorSpacingX", m_fDetectorSpacingX);
	_sNode->addChildNode("DetectorSpacingY", m_fDetectorSpacingY);
	_sNode->addChildNode("DetectorRowCount", m_iDetectorRowCount);
	_sNode->addChildNode("DetectorColCount", m_iDetectorColCount);
	_sNode->addChildNode("ProjectionAngles", m_pfProjectionAngles, m_iProjectionAngleCount);
	_sNode->addChildNode("DistanceOriginDetector", m_fOriginDetectorDistance);
	_sNode->addChildNode("DistanceOriginSource", m_fOriginSourceDistance);
}
//----------------------------------------------------------------------------------------

CVector3D CConeProjectionGeometry3D::getProjectionDirection(int _iProjectionIndex, int _iDetectorIndex) const
{
	float32 fSrcX = -m_fOriginSourceDistance;
	float32 fSrcY = 0.0f;
	float32 fSrcZ = 0.0f;

	float32 fDetX = m_fOriginDetectorDistance;
	float32 fDetY = 0.0f;
	float32 fDetZ = 0.0f;

	fDetY += indexToDetectorOffsetX(_iDetectorIndex);
	fDetZ += indexToDetectorOffsetY(_iDetectorIndex);

	float32 angle = m_pfProjectionAngles[_iProjectionIndex];

	#define ROTATE(name,alpha) do { float32 tX = f##name##X * cos(alpha) - f##name##Y * sin(alpha); f##name##Y = f##name##X * sin(alpha) + f##name##Y * cos(alpha); f##name##X = tX; } while(0)

	ROTATE(Src, angle);
	ROTATE(Det, angle);

	#undef ROTATE

	CVector3D ret(fDetX - fSrcX, fDetY - fSrcY, fDetZ - fDetZ);
	return ret;
}

void CConeProjectionGeometry3D::projectPoint(float32 fX, float32 fY, float32 fZ,
                                                 int iAngleIndex,
                                                 float32 &fU, float32 &fV) const
{
	ASTRA_ASSERT(iAngleIndex >= 0);
	ASTRA_ASSERT(iAngleIndex < m_iProjectionAngleCount);

	float alpha = m_pfProjectionAngles[iAngleIndex];

	// Project point onto optical axis

	// Projector direction is (cos(alpha), sin(alpha))
	// Vector source->origin is (-sin(alpha), cos(alpha))

	// Distance from source, projected on optical axis
	float fD = -sin(alpha) * fX + cos(alpha) * fY + m_fOriginSourceDistance;

	// Scale fZ to detector plane
	fV = detectorOffsetYToRowIndexFloat( (fZ * (m_fOriginSourceDistance + m_fOriginDetectorDistance)) / fD );


	// Orthogonal distance in XY-plane to optical axis
	float fS = cos(alpha) * fX + sin(alpha) * fY;

	// Scale fS to detector plane
	fU = detectorOffsetXToColIndexFloat( (fS * (m_fOriginSourceDistance + m_fOriginDetectorDistance)) / fD );

	fprintf(stderr, "alpha: %f, D: %f, V: %f, S: %f, U: %f\n", alpha, fD, fV, fS, fU);

}


} // end namespace astra
