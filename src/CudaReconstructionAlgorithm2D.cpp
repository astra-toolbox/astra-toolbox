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

#ifdef ASTRA_CUDA

#include "astra/CudaReconstructionAlgorithm2D.h"

#include <boost/lexical_cast.hpp>

#include "astra/AstraObjectManager.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/CudaProjector2D.h"

#include "../cuda/2d/algo.h"

#include <ctime>

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Constructor
CCudaReconstructionAlgorithm2D::CCudaReconstructionAlgorithm2D() 
{
	_clear();
}



//----------------------------------------------------------------------------------------
// Destructor
CCudaReconstructionAlgorithm2D::~CCudaReconstructionAlgorithm2D() 
{
	delete m_pAlgo;
	m_pAlgo = 0;
	m_bAlgoInit = false;
}

void CCudaReconstructionAlgorithm2D::clear()
{
	delete m_pAlgo;
	_clear();
}

void CCudaReconstructionAlgorithm2D::_clear()
{
	m_bIsInitialized = false;
	m_pAlgo = 0;
	m_bAlgoInit = false;
	CReconstructionAlgorithm2D::_clear();

	m_iGPUIndex = 0;
	m_iDetectorSuperSampling = 1;
	m_iPixelSuperSampling = 1;
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaReconstructionAlgorithm2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaReconstructionAlgorithm2D", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// sinogram data
	XMLNode* node = _cfg.self->getSingleNode("ProjectionDataId");
	ASTRA_CONFIG_CHECK(node, "CudaSirt2", "No ProjectionDataId tag specified.");
	int id = boost::lexical_cast<int>(node->getContent());
	m_pSinogram = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	ASTRA_DELETE(node);
	CC.markNodeParsed("ProjectionDataId");

	// reconstruction data
	node = _cfg.self->getSingleNode("ReconstructionDataId");
	ASTRA_CONFIG_CHECK(node, "CudaSirt2", "No ReconstructionDataId tag specified.");
	id = boost::lexical_cast<int>(node->getContent());
	m_pReconstruction = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	ASTRA_DELETE(node);
	CC.markNodeParsed("ReconstructionDataId");

	// fixed mask
	if (_cfg.self->hasOption("ReconstructionMaskId")) {
		m_bUseReconstructionMask = true;
		id = boost::lexical_cast<int>(_cfg.self->getOption("ReconstructionMaskId"));
		m_pReconstructionMask = dynamic_cast<CFloat32VolumeData2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("ReconstructionMaskId");
	// fixed mask
	if (_cfg.self->hasOption("SinogramMaskId")) {
		m_bUseSinogramMask = true;
		id = boost::lexical_cast<int>(_cfg.self->getOption("SinogramMaskId"));
		m_pSinogramMask = dynamic_cast<CFloat32ProjectionData2D*>(CData2DManager::getSingleton().get(id));
	}
	CC.markOptionParsed("SinogramMaskId");

	// Constraints - NEW
	if (_cfg.self->hasOption("MinConstraint")) {
		m_bUseMinConstraint = true;
		m_fMinValue = _cfg.self->getOptionNumerical("MinConstraint", 0.0f);
		CC.markOptionParsed("MinConstraint");
	} else {
		// Constraint - OLD
		m_bUseMinConstraint = _cfg.self->getOptionBool("UseMinConstraint", false);
		CC.markOptionParsed("UseMinConstraint");
		if (m_bUseMinConstraint) {
			m_fMinValue = _cfg.self->getOptionNumerical("MinConstraintValue", 0.0f);
			CC.markOptionParsed("MinConstraintValue");
		}
	}
	if (_cfg.self->hasOption("MaxConstraint")) {
		m_bUseMaxConstraint = true;
		m_fMaxValue = _cfg.self->getOptionNumerical("MaxConstraint", 255.0f);
		CC.markOptionParsed("MaxConstraint");
	} else {
		// Constraint - OLD
		m_bUseMaxConstraint = _cfg.self->getOptionBool("UseMaxConstraint", false);
		CC.markOptionParsed("UseMaxConstraint");
		if (m_bUseMaxConstraint) {
			m_fMaxValue = _cfg.self->getOptionNumerical("MaxConstraintValue", 0.0f);
			CC.markOptionParsed("MaxConstraintValue");
		}
	}

	// GPU number
	m_iGPUIndex = (int)_cfg.self->getOptionNumerical("GPUindex", 0);
	m_iGPUIndex = (int)_cfg.self->getOptionNumerical("GPUIndex", m_iGPUIndex);
	CC.markOptionParsed("GPUindex");
	if (!_cfg.self->hasOption("GPUindex"))
		CC.markOptionParsed("GPUIndex");

	// Detector supersampling factor
	m_iDetectorSuperSampling = (int)_cfg.self->getOptionNumerical("DetectorSuperSampling", 1);
	CC.markOptionParsed("DetectorSuperSampling");

	// Pixel supersampling factor
	m_iPixelSuperSampling = (int)_cfg.self->getOptionNumerical("PixelSuperSampling", 1);
	CC.markOptionParsed("PixelSuperSampling");


	// This isn't used yet, but passing it is not something to warn about
	node = _cfg.self->getSingleNode("ProjectorId");
	if (node) {
		id = boost::lexical_cast<int>(node->getContent());
		CProjector2D *projector = CProjector2DManager::getSingleton().get(id);
		if (!dynamic_cast<CCudaProjector2D*>(projector)) {
			cout << "Warning: non-CUDA Projector2D passed to FP_CUDA" << std::endl;
		}
		delete node;
	}
	CC.markNodeParsed("ProjectorId");


	return _check();
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaReconstructionAlgorithm2D::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram, 
                                     CFloat32VolumeData2D* _pReconstruction)
{
	return initialize(_pProjector, _pSinogram, _pReconstruction, 0, 1);
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaReconstructionAlgorithm2D::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram, 
                                     CFloat32VolumeData2D* _pReconstruction,
                                     int _iGPUindex,
                                     int _iDetectorSuperSampling,
                                     int _iPixelSuperSampling)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}
	
	m_pProjector = 0;
	
	// required classes
	m_pSinogram = _pSinogram;
	m_pReconstruction = _pReconstruction;

	m_iDetectorSuperSampling = _iDetectorSuperSampling;
	m_iPixelSuperSampling = _iPixelSuperSampling;
	m_iGPUIndex = _iGPUindex;

	return _check();
}


//----------------------------------------------------------------------------------------
// Check
bool CCudaReconstructionAlgorithm2D::_check() 
{
	// TODO: CLEAN UP


	// check pointers
	//ASTRA_CONFIG_CHECK(m_pProjector, "Reconstruction2D", "Invalid Projector Object.");
	ASTRA_CONFIG_CHECK(m_pSinogram, "SIRT_CUDA", "Invalid Projection Data Object.");
	ASTRA_CONFIG_CHECK(m_pReconstruction, "SIRT_CUDA", "Invalid Reconstruction Data Object.");

	// check initializations
	//ASTRA_CONFIG_CHECK(m_pProjector->isInitialized(), "Reconstruction2D", "Projector Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pSinogram->isInitialized(), "SIRT_CUDA", "Projection Data Object Not Initialized.");
	ASTRA_CONFIG_CHECK(m_pReconstruction->isInitialized(), "SIRT_CUDA", "Reconstruction Data Object Not Initialized.");

	ASTRA_CONFIG_CHECK(m_iDetectorSuperSampling >= 1, "SIRT_CUDA", "DetectorSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iPixelSuperSampling >= 1, "SIRT_CUDA", "PixelSuperSampling must be a positive integer.");
	ASTRA_CONFIG_CHECK(m_iGPUIndex >= 0, "SIRT_CUDA", "GPUIndex must be a non-negative integer.");

	// check compatibility between projector and data classes
//	ASTRA_CONFIG_CHECK(m_pSinogram->getGeometry()->isEqual(m_pProjector->getProjectionGeometry()), "SIRT_CUDA", "Projection Data not compatible with the specified Projector.");
//	ASTRA_CONFIG_CHECK(m_pReconstruction->getGeometry()->isEqual(m_pProjector->getVolumeGeometry()), "SIRT_CUDA", "Reconstruction Data not compatible with the specified Projector.");

	// todo: turn some of these back on

// 	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "SIRT_CUDA", "ProjectionGeometry not specified.");
// 	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "SIRT_CUDA", "ProjectionGeometry not initialized.");
// 	ASTRA_CONFIG_CHECK(m_pReconstructionGeometry, "SIRT_CUDA", "ReconstructionGeometry not specified.");
// 	ASTRA_CONFIG_CHECK(m_pReconstructionGeometry->isInitialized(), "SIRT_CUDA", "ReconstructionGeometry not initialized.");

	// check dimensions
	//ASTRA_CONFIG_CHECK(m_pSinogram->getAngleCount() == m_pProjectionGeometry->getProjectionAngleCount(), "SIRT_CUDA", "Sinogram data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pSinogram->getDetectorCount() == m_pProjectionGeometry->getDetectorCount(), "SIRT_CUDA", "Sinogram data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pReconstruction->getWidth() == m_pReconstructionGeometry->getGridColCount(), "SIRT_CUDA", "Reconstruction data object size mismatch.");
	//ASTRA_CONFIG_CHECK(m_pReconstruction->getHeight() == m_pReconstructionGeometry->getGridRowCount(), "SIRT_CUDA", "Reconstruction data object size mismatch.");
	
	// check restrictions
	// TODO: check restrictions built into cuda code


	// success
	m_bIsInitialized = true;
	return true;
}

void CCudaReconstructionAlgorithm2D::setGPUIndex(int _iGPUIndex)
{
	m_iGPUIndex = _iGPUIndex;
}


//---------------------------------------------------------------------------------------
// Information - All
map<string,boost::any> CCudaReconstructionAlgorithm2D::getInformation()
{
	// TODO: Verify and clean up

	map<string,boost::any> res;
	res["ProjectionGeometry"] = getInformation("ProjectionGeometry");
	res["ReconstructionGeometry"] = getInformation("ReconstructionGeometry");
	res["ProjectionDataId"] = getInformation("ProjectionDataId");
	res["ReconstructionDataId"] = getInformation("ReconstructionDataId");
	res["ReconstructionMaskId"] = getInformation("ReconstructionMaskId");
	res["GPUindex"] = getInformation("GPUindex");
	res["DetectorSuperSampling"] = getInformation("DetectorSuperSampling");
	res["PixelSuperSampling"] = getInformation("PixelSuperSampling");
	res["UseMinConstraint"] = getInformation("UseMinConstraint");
	res["MinConstraintValue"] = getInformation("MinConstraintValue");
	res["UseMaxConstraint"] = getInformation("UseMaxConstraint");
	res["MaxConstraintValue"] = getInformation("MaxConstraintValue");
	return mergeMap<string,boost::any>(CReconstructionAlgorithm2D::getInformation(), res);
}

//---------------------------------------------------------------------------------------
// Information - Specific
boost::any CCudaReconstructionAlgorithm2D::getInformation(std::string _sIdentifier)
{
	// TODO: Verify and clean up

	if (_sIdentifier == "UseMinConstraint")		{ return m_bUseMinConstraint ? string("yes") : string("no"); }
	if (_sIdentifier == "MinConstraintValue")	{ return m_fMinValue; }
	if (_sIdentifier == "UseMaxConstraint")		{ return m_bUseMaxConstraint ? string("yes") : string("no"); }
	if (_sIdentifier == "MaxConstraintValue")	{ return m_fMaxValue; }

	// TODO: store these so we can return them?
	if (_sIdentifier == "ProjectionGeometry")	{ return string("not implemented"); }
	if (_sIdentifier == "ReconstructionGeometry")	{ return string("not implemented"); }
	if (_sIdentifier == "GPUindex")	{ return m_iGPUIndex; }
	if (_sIdentifier == "DetectorSuperSampling")	{ return m_iDetectorSuperSampling; }
	if (_sIdentifier == "PixelSuperSampling")	{ return m_iPixelSuperSampling; }

	if (_sIdentifier == "ProjectionDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pSinogram);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	if (_sIdentifier == "ReconstructionDataId") {
		int iIndex = CData2DManager::getSingleton().getIndex(m_pReconstruction);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	if (_sIdentifier == "ReconstructionMaskId") {
		if (!m_bUseReconstructionMask) return string("not used");
		int iIndex = CData2DManager::getSingleton().getIndex(m_pReconstructionMask);
		if (iIndex != 0) return iIndex;
		return std::string("not in manager");
	}
	return CReconstructionAlgorithm2D::getInformation(_sIdentifier);
}

bool CCudaReconstructionAlgorithm2D::setupGeometry()
{
	ASTRA_ASSERT(m_bIsInitialized);
	ASTRA_ASSERT(!m_bAlgoInit);

	bool ok;

	// TODO: Probably not the best place for this...
	ok = m_pAlgo->setGPUIndex(m_iGPUIndex);
	if (!ok) return false;

	astraCUDA::SDimensions dims;

	const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();

	// TODO: off-center geometry, non-square pixels
	dims.iVolWidth = volgeom.getGridColCount();
	dims.iVolHeight = volgeom.getGridRowCount();
	float fPixelSize = volgeom.getPixelLengthX();

	dims.iRaysPerDet = m_iDetectorSuperSampling;
	dims.iRaysPerPixelDim = m_iPixelSuperSampling;


	const CParallelProjectionGeometry2D* parProjGeom = dynamic_cast<CParallelProjectionGeometry2D*>(m_pSinogram->getGeometry());
	const CFanFlatProjectionGeometry2D* fanProjGeom = dynamic_cast<CFanFlatProjectionGeometry2D*>(m_pSinogram->getGeometry());
	const CFanFlatVecProjectionGeometry2D* fanVecProjGeom = dynamic_cast<CFanFlatVecProjectionGeometry2D*>(m_pSinogram->getGeometry());

	if (parProjGeom) {

		dims.iProjAngles = parProjGeom->getProjectionAngleCount();
		dims.iProjDets = parProjGeom->getDetectorCount();
		dims.fDetScale = parProjGeom->getDetectorWidth() / fPixelSize;

		ok = m_pAlgo->setGeometry(dims, parProjGeom->getProjectionAngles());

	} else if (fanProjGeom) {

		dims.iProjAngles = fanProjGeom->getProjectionAngleCount();
		dims.iProjDets = fanProjGeom->getDetectorCount();
		dims.fDetScale = fanProjGeom->getDetectorWidth() / fPixelSize;
		float fOriginSourceDistance = fanProjGeom->getOriginSourceDistance();
		float fOriginDetectorDistance = fanProjGeom->getOriginDetectorDistance();

		const float* angles = fanProjGeom->getProjectionAngles();

		astraCUDA::SFanProjection* projs;
		projs = new astraCUDA::SFanProjection[dims.iProjAngles];

		float fSrcX0 = 0.0f;
		float fSrcY0 = -fOriginSourceDistance / fPixelSize;
		float fDetUX0 = dims.fDetScale;
		float fDetUY0 = 0.0f;
		float fDetSX0 = dims.iProjDets * fDetUX0 / -2.0f;
		float fDetSY0 = fOriginDetectorDistance / fPixelSize;

#define ROTATE0(name,i,alpha) do { projs[i].f##name##X = f##name##X0 * cos(alpha) - f##name##Y0 * sin(alpha); projs[i].f##name##Y = f##name##X0 * sin(alpha) + f##name##Y0 * cos(alpha); } while(0)
		for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
			ROTATE0(Src, i, angles[i]);
			ROTATE0(DetS, i, angles[i]);
			ROTATE0(DetU, i, angles[i]);
		}

#undef ROTATE0

		ok = m_pAlgo->setFanGeometry(dims, projs);
		delete[] projs;

	} else if (fanVecProjGeom) {

		dims.iProjAngles = fanVecProjGeom->getProjectionAngleCount();
		dims.iProjDets = fanVecProjGeom->getDetectorCount();
		dims.fDetScale = fanVecProjGeom->getDetectorWidth() / fPixelSize;

		const astraCUDA::SFanProjection* projs;
		projs = fanVecProjGeom->getProjectionVectors();

		// Rescale projs to fPixelSize == 1

		astraCUDA::SFanProjection* scaledProjs = new astraCUDA::SFanProjection[dims.iProjAngles];
#define SCALE(name,i,alpha) do { scaledProjs[i].f##name##X = projs[i].f##name##X * alpha; scaledProjs[i].f##name##Y = projs[i].f##name##Y * alpha; } while (0)
		for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
			SCALE(Src,i,1.0f/fPixelSize);
			SCALE(DetS,i,1.0f/fPixelSize);
			SCALE(DetU,i,1.0f/fPixelSize);
		}

		ok = m_pAlgo->setFanGeometry(dims, scaledProjs);

		delete[] scaledProjs;

	} else {

		ASTRA_ASSERT(false);

	}
	if (!ok) return false;


	if (m_bUseReconstructionMask)
		ok &= m_pAlgo->enableVolumeMask();
	if (!ok) return false;
	if (m_bUseSinogramMask)
		ok &= m_pAlgo->enableSinogramMask();
	if (!ok) return false;

	const float *pfTOffsets = m_pSinogram->getGeometry()->getExtraDetectorOffset();
	if (pfTOffsets)
		ok &= m_pAlgo->setTOffsets(pfTOffsets);
	if (!ok) return false;

	ok &= m_pAlgo->init();
	if (!ok) return false;


	return true;
}

//----------------------------------------------------------------------------------------
// Iterate
void CCudaReconstructionAlgorithm2D::run(int _iNrIterations)
{
	// check initialized
	ASTRA_ASSERT(m_bIsInitialized);

	bool ok = true;
	const CVolumeGeometry2D& volgeom = *m_pReconstruction->getGeometry();

	if (!m_bAlgoInit) {

		ok = setupGeometry();
		ASTRA_ASSERT(ok);

		ok = m_pAlgo->allocateBuffers();
		ASTRA_ASSERT(ok);

		m_bAlgoInit = true;
	}

	float fPixelSize = volgeom.getPixelLengthX();
	float fSinogramScale = 1.0f/(fPixelSize*fPixelSize);

	ok = m_pAlgo->copyDataToGPU(m_pSinogram->getDataConst(), m_pSinogram->getGeometry()->getDetectorCount(), fSinogramScale,
	                            m_pReconstruction->getDataConst(), volgeom.getGridColCount(),
	                            m_bUseReconstructionMask ? m_pReconstructionMask->getDataConst() : 0, volgeom.getGridColCount(),
	                            m_bUseSinogramMask ? m_pSinogramMask->getDataConst() : 0, m_pSinogram->getGeometry()->getDetectorCount());

	ASTRA_ASSERT(ok);

	if (m_bUseMinConstraint)
		ok &= m_pAlgo->setMinConstraint(m_fMinValue);
	if (m_bUseMaxConstraint)
		ok &= m_pAlgo->setMaxConstraint(m_fMaxValue);

	ok &= m_pAlgo->iterate(_iNrIterations);
	ASTRA_ASSERT(ok);

	ok &= m_pAlgo->getReconstruction(m_pReconstruction->getData(),
	                                 volgeom.getGridColCount());

	ASTRA_ASSERT(ok);
}

void CCudaReconstructionAlgorithm2D::signalAbort()
{
	if (m_bIsInitialized && m_pAlgo) {
		m_pAlgo->signalAbort();
	}
}

bool CCudaReconstructionAlgorithm2D::getResidualNorm(float32& _fNorm)
{
	if (!m_bIsInitialized || !m_pAlgo)
		return false;

	_fNorm = m_pAlgo->computeDiffNorm();

	return true;
}

} // namespace astra

#endif // ASTRA_CUDA
