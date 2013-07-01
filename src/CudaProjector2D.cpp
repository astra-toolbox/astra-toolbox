/*
-----------------------------------------------------------------------
This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA-Toolbox")

Copyright: IBBT-Vision Lab, University of Antwerp
Contact: mailto:wim.vanaarle@ua.ac.be
Website: http://astra.ua.ac.be
-----------------------------------------------------------------------
$Id$
*/
#include "astra/CudaProjector2D.h"


namespace astra
{

// type of the projector, needed to register with CProjectorFactory
std::string CCudaProjector2D::type = "cuda";


//----------------------------------------------------------------------------------------
// Default constructor
CCudaProjector2D::CCudaProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaProjector2D::~CCudaProjector2D()
{
	if (m_bIsInitialized) clear();
}

//----------------------------------------------------------------------------------------
// Clear for constructors
void CCudaProjector2D::_clear()
{
	m_pProjectionGeometry = NULL;
	m_pVolumeGeometry = NULL;
	m_bIsInitialized = false;

	m_projectionKernel = ker2d_default;
}

//----------------------------------------------------------------------------------------
// Clear
void CCudaProjector2D::clear()
{
	ASTRA_DELETE(m_pProjectionGeometry);
	ASTRA_DELETE(m_pVolumeGeometry);
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaProjector2D::_check()
{
	// projection geometry
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "CudaProjector2D", "ProjectionGeometry2D not initialized.");
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "CudaProjector2D", "ProjectionGeometry2D not initialized.");

	// volume geometry
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry, "CudaProjector2D", "VolumeGeometry2D not initialized.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "CudaProjector2D", "VolumeGeometry2D not initialized.");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CCudaProjector2D::initialize(const Config& _cfg)
{
	assert(_cfg.self);
	ConfigStackCheck<CProjector2D> CC("CudaProjector2D", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector2D::initialize(_cfg)) {
		return false;
	}

	// TODO: Check the projection geometry is a supported type

	XMLNode* node = _cfg.self->getSingleNode("ProjectionKernel");
	m_projectionKernel = ker2d_default;
	if (node) {
		std::string sProjKernel = node->getContent();

		if (sProjKernel == "default") {

		} else {
			return false;
		}
	}
	ASTRA_DELETE(node);
	CC.markNodeParsed("ProjectionKernel");

	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

/*
bool CProjector2D::initialize(astra::CProjectionGeometry2D *, astra::CVolumeGeometry2D *)
{
	ASTRA_ASSERT(false);

	return false;
}
*/

std::string CCudaProjector2D::description() const
{
	return "";
}

} // end namespace
