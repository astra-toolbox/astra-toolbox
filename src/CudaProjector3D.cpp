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

#include "astra/CudaProjector3D.h"

#include "astra/VolumeGeometry3D.h"
#include "astra/ProjectionGeometry3D.h"

#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"

#include "astra/Logging.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor
CCudaProjector3D::CCudaProjector3D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaProjector3D::~CCudaProjector3D()
{
	if (m_bIsInitialized) clear();
}

//----------------------------------------------------------------------------------------
// Clear for constructors
void CCudaProjector3D::_clear()
{
	m_pProjectionGeometry.reset();
	m_pVolumeGeometry.reset();
	m_bIsInitialized = false;

	m_projectionKernel = ker3d_default;
	m_iVoxelSuperSampling = 1;
	m_iDetectorSuperSampling = 1;
	m_iGPUIndex = -1;
}

//----------------------------------------------------------------------------------------
// Clear
void CCudaProjector3D::clear()
{
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check
bool CCudaProjector3D::_check()
{
	// projection geometry
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "CudaProjector3D", "ProjectionGeometry3D not initialized.");
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "CudaProjector3D", "ProjectionGeometry3D not initialized.");

	// volume geometry
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry, "CudaProjector3D", "VolumeGeometry3D not initialized.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "CudaProjector3D", "VolumeGeometry3D not initialized.");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CCudaProjector3D::initialize(const Config& _cfg)
{
	ConfigReader<CProjector3D> CR("CudaProjector3D", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector3D::initialize(_cfg)) {
		return false;
	}

	std::string sProjKernel;
	CR.getString("ProjectionKernel", sProjKernel, "default");
	if (sProjKernel == "default") {
		m_projectionKernel = ker3d_default;
	} else if (sProjKernel == "sum_square_weights") {
		m_projectionKernel = ker3d_sum_square_weights;
	} else if (sProjKernel == "2d_weighting") {
		m_projectionKernel = ker3d_2d_weighting;
	} else {
		ASTRA_ERROR("Unknown ProjectionKernel");
		return false;
	}

	bool ok = true;

	ok &= CR.getOptionInt("VoxelSuperSampling", m_iVoxelSuperSampling, 1);
	ok &= CR.getOptionInt("DetectorSuperSampling", m_iDetectorSuperSampling, 1);

	if (CR.hasOption("GPUIndex"))
		ok &= CR.getOptionInt("GPUIndex", m_iGPUIndex, -1);
	else
		ok &= CR.getOptionInt("GPUindex", m_iGPUIndex, -1);
	if (!ok)
		return false;

	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

/*
bool CProjector3D::initialize(astra::CProjectionGeometry3D *, astra::CVolumeGeometry3D *)
{
	ASTRA_ASSERT(false);

	return false;
}
*/

std::string CCudaProjector3D::description() const
{
	return "";
}

} // end namespace
