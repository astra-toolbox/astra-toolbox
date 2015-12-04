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
*/

#include "astra/CompositeGeometryManager.h"

#ifdef ASTRA_CUDA

#include "astra/GeometryUtil3D.h"
#include "astra/VolumeGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/Projector3D.h"
#include "astra/CudaProjector3D.h"
#include "astra/Float32ProjectionData3DMemory.h"
#include "astra/Float32VolumeData3DMemory.h"
#include "astra/Logging.h"

#include "../cuda/3d/mem3d.h"

#include <cstring>

namespace astra {

// JOB:
//  
// VolumePart
// ProjectionPart
// FP-or-BP
// SET-or-ADD


// Running a set of jobs:
//
// [ Assume OUTPUT Parts in a single JobSet don't alias?? ]
// Group jobs by output Part
// One thread per group?

// Automatically split parts if too large
// Performance model for odd-sized tasks?
// Automatically split parts if not enough tasks to fill available GPUs


// Splitting:
// Constraints:
//   number of sub-parts divisible by N
//   max size of sub-parts

// For splitting on both input and output side:
//   How to divide up memory? (Optimization problem; compute/benchmark)
//   (First approach: 0.5/0.5)



bool CCompositeGeometryManager::splitJobs(TJobSet &jobs, size_t maxSize, int div, TJobSet &split)
{
	split.clear();

	for (TJobSet::const_iterator i = jobs.begin(); i != jobs.end(); ++i)
	{
		CPart* pOutput = i->first;
		const TJobList &L = i->second;

		// 1. Split output part
		// 2. Per sub-part:
		//    a. reduce input part
		//    b. split input part
		//    c. create jobs for new (input,output) subparts

		TPartList splitOutput = pOutput->split(maxSize/3, div);

		for (TJobList::const_iterator j = L.begin(); j != L.end(); ++j)
		{
			const SJob &job = *j;

			for (TPartList::iterator i_out = splitOutput.begin();
			     i_out != splitOutput.end(); ++i_out)
			{
				boost::shared_ptr<CPart> outputPart = *i_out;

				SJob newjob;
				newjob.pOutput = outputPart;
				newjob.eType = j->eType;
				newjob.eMode = j->eMode;
				newjob.pProjector = j->pProjector;

				CPart* input = job.pInput->reduce(outputPart.get());

				if (input->getSize() == 0) {
					ASTRA_DEBUG("Empty input");
					newjob.eType = SJob::JOB_NOP;
					split[outputPart.get()].push_back(newjob);
					continue;
				}

				size_t remainingSize = ( maxSize - outputPart->getSize() ) / 2;

				TPartList splitInput = input->split(remainingSize, 1);
				delete input;
				ASTRA_DEBUG("Input split into %d parts", splitInput.size());

				for (TPartList::iterator i_in = splitInput.begin();
				     i_in != splitInput.end(); ++i_in)
				{
					newjob.pInput = *i_in;

					split[outputPart.get()].push_back(newjob);

					// Second and later (input) parts should always be added to
					// output of first (input) part.
					newjob.eMode = SJob::MODE_ADD;
				}

			
			}

		}
	}

	return true;
}

CCompositeGeometryManager::CPart::CPart(const CPart& other)
{
	eType = other.eType;
	pData = other.pData;
	subX = other.subX;
	subY = other.subY;
	subZ = other.subZ;
}

CCompositeGeometryManager::CVolumePart::CVolumePart(const CVolumePart& other)
 : CPart(other)
{
	pGeom = other.pGeom->clone();
}

CCompositeGeometryManager::CVolumePart::~CVolumePart()
{
	delete pGeom;
}

void CCompositeGeometryManager::CVolumePart::getDims(size_t &x, size_t &y, size_t &z)
{
	if (!pGeom) {
		x = y = z = 0;
		return;
	}

	x = pGeom->getGridColCount();
	y = pGeom->getGridRowCount();
	z = pGeom->getGridSliceCount();
}

size_t CCompositeGeometryManager::CPart::getSize()
{
	size_t x, y, z;
	getDims(x, y, z);
	return x * y * z;
}



CCompositeGeometryManager::CPart* CCompositeGeometryManager::CVolumePart::reduce(const CPart *_other)
{
	const CProjectionPart *other = dynamic_cast<const CProjectionPart *>(_other);
	assert(other);

	// TODO: Is 0.5 sufficient?
	double umin = -0.5;
	double umax = other->pGeom->getDetectorColCount() + 0.5;
	double vmin = -0.5;
	double vmax = other->pGeom->getDetectorRowCount() + 0.5;

	double uu[4];
	double vv[4];
	uu[0] = umin; vv[0] = vmin;
	uu[1] = umin; vv[1] = vmax;
	uu[2] = umax; vv[2] = vmin;
	uu[3] = umax; vv[3] = vmax;

	double pixx = pGeom->getPixelLengthX();
	double pixy = pGeom->getPixelLengthY();
	double pixz = pGeom->getPixelLengthZ();

	double xmin = pGeom->getWindowMinX() - 0.5 * pixx;
	double xmax = pGeom->getWindowMaxX() + 0.5 * pixx;
	double ymin = pGeom->getWindowMinY() - 0.5 * pixy;
	double ymax = pGeom->getWindowMaxY() + 0.5 * pixy;

	// NB: Flipped
	double zmax = pGeom->getWindowMinZ() - 2.5 * pixz;
	double zmin = pGeom->getWindowMaxZ() + 2.5 * pixz;

	// TODO: This isn't as tight as it could be.
	// In particular it won't detect the detector being
	// missed entirely on the u side.

	for (int i = 0; i < other->pGeom->getProjectionCount(); ++i) {
		for (int j = 0; j < 4; ++j) {
			double px, py, pz;

			other->pGeom->backprojectPointX(i, uu[j], vv[j], xmin, py, pz);
			//ASTRA_DEBUG("%f %f (%f - %f)", py, pz, ymin, ymax);
			if (pz < zmin) zmin = pz;
			if (pz > zmax) zmax = pz;
			other->pGeom->backprojectPointX(i, uu[j], vv[j], xmax, py, pz);
			//ASTRA_DEBUG("%f %f (%f - %f)", py, pz, ymin, ymax);
			if (pz < zmin) zmin = pz;
			if (pz > zmax) zmax = pz;

			other->pGeom->backprojectPointY(i, uu[j], vv[j], ymin, px, pz);
			//ASTRA_DEBUG("%f %f (%f - %f)", px, pz, xmin, xmax);
			if (pz < zmin) zmin = pz;
			if (pz > zmax) zmax = pz;
			other->pGeom->backprojectPointY(i, uu[j], vv[j], ymax, px, pz);
			//ASTRA_DEBUG("%f %f (%f - %f)", px, pz, xmin, xmax);
			if (pz < zmin) zmin = pz;
			if (pz > zmax) zmax = pz;
		}
	}

	//ASTRA_DEBUG("coord extent: %f - %f", zmin, zmax);

	zmin = (zmin - pixz - pGeom->getWindowMinZ()) / pixz;
	zmax = (zmax + pixz - pGeom->getWindowMinZ()) / pixz;

	int _zmin = (int)floor(zmin);
	int _zmax = (int)ceil(zmax);

	//ASTRA_DEBUG("index extent: %d - %d", _zmin, _zmax);

	if (_zmin < 0)
		_zmin = 0;
	if (_zmax > pGeom->getGridSliceCount())
		_zmax = pGeom->getGridSliceCount();

	if (_zmax <= _zmin) {
		_zmin = _zmax = 0;
	}
	//ASTRA_DEBUG("adjusted extent: %d - %d", _zmin, _zmax);

	CVolumePart *sub = new CVolumePart();
	sub->subX = this->subX;
	sub->subY = this->subY;
	sub->subZ = this->subZ + _zmin;
	sub->pData = pData;

	if (_zmin == _zmax) {
		sub->pGeom = 0;
	} else {
		sub->pGeom = new CVolumeGeometry3D(pGeom->getGridColCount(),
		                                   pGeom->getGridRowCount(),
		                                   _zmax - _zmin,
		                                   pGeom->getWindowMinX(),
		                                   pGeom->getWindowMinY(),
		                                   pGeom->getWindowMinZ() + _zmin * pixz,
		                                   pGeom->getWindowMaxX(),
		                                   pGeom->getWindowMaxY(),
		                                   pGeom->getWindowMinZ() + _zmax * pixz);
	}

	ASTRA_DEBUG("Reduce volume from %d - %d to %d - %d", this->subZ, this->subZ +  pGeom->getGridSliceCount(), this->subZ + _zmin, this->subZ + _zmax);

	return sub;
}



static size_t ceildiv(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static size_t computeVerticalSplit(size_t maxBlock, int div, size_t sliceCount)
{
    size_t blockSize = maxBlock;
    size_t blockCount = ceildiv(sliceCount, blockSize);

    // Increase number of blocks to be divisible by div
    size_t divCount = div * ceildiv(blockCount, div);

    // If divCount is above sqrt(number of slices), then
    // we can't guarantee divisibility by div, but let's try anyway
    if (ceildiv(sliceCount, ceildiv(sliceCount, divCount)) % div == 0) {
        blockCount = divCount;
    } else {
        // If divisibility isn't achievable, we may want to optimize
        // differently.
        // TODO: Figure out how to model and optimize this.
    }

    // Final adjustment to make blocks more evenly sized
    // (This can't make the blocks larger)
    blockSize = ceildiv(sliceCount, blockCount); 

    ASTRA_DEBUG("%ld %ld -> %ld * %ld", sliceCount, maxBlock, blockCount, blockSize);

    assert(blockSize <= maxBlock);
    assert((divCount * divCount > sliceCount) || (blockCount % div) == 0);

    return blockSize;
}

template<class V, class P>
static V* getProjectionVectors(const P* geom);

template<>
SConeProjection* getProjectionVectors(const CConeProjectionGeometry3D* pProjGeom)
{
	return genConeProjections(pProjGeom->getProjectionCount(),
	                          pProjGeom->getDetectorColCount(),
	                          pProjGeom->getDetectorRowCount(),
	                          pProjGeom->getOriginSourceDistance(),
	                          pProjGeom->getOriginDetectorDistance(),
	                          pProjGeom->getDetectorSpacingX(),
	                          pProjGeom->getDetectorSpacingY(),
	                          pProjGeom->getProjectionAngles());
}

template<>
SConeProjection* getProjectionVectors(const CConeVecProjectionGeometry3D* pProjGeom)
{
	int nth = pProjGeom->getProjectionCount();

	SConeProjection* pProjs = new SConeProjection[nth];
	for (int i = 0; i < nth; ++i)
		pProjs[i] = pProjGeom->getProjectionVectors()[i];

	return pProjs;
}

template<>
SPar3DProjection* getProjectionVectors(const CParallelProjectionGeometry3D* pProjGeom)
{
	return genPar3DProjections(pProjGeom->getProjectionCount(),
	                           pProjGeom->getDetectorColCount(),
	                           pProjGeom->getDetectorRowCount(),
	                           pProjGeom->getDetectorSpacingX(),
	                           pProjGeom->getDetectorSpacingY(),
	                           pProjGeom->getProjectionAngles());
}

template<>
SPar3DProjection* getProjectionVectors(const CParallelVecProjectionGeometry3D* pProjGeom)
{
	int nth = pProjGeom->getProjectionCount();

	SPar3DProjection* pProjs = new SPar3DProjection[nth];
	for (int i = 0; i < nth; ++i)
		pProjs[i] = pProjGeom->getProjectionVectors()[i];

	return pProjs;
}


template<class V>
static void translateProjectionVectors(V* pProjs, int count, double dv)
{
	for (int i = 0; i < count; ++i) {
		pProjs[i].fDetSX += dv * pProjs[i].fDetVX;
		pProjs[i].fDetSY += dv * pProjs[i].fDetVY;
		pProjs[i].fDetSZ += dv * pProjs[i].fDetVZ;
	}
}



static CProjectionGeometry3D* getSubProjectionGeometry(const CProjectionGeometry3D* pProjGeom, int v, int size)
{
	// First convert to vectors, then translate, then convert into new object

	const CConeProjectionGeometry3D* conegeom = dynamic_cast<const CConeProjectionGeometry3D*>(pProjGeom);
	const CParallelProjectionGeometry3D* par3dgeom = dynamic_cast<const CParallelProjectionGeometry3D*>(pProjGeom);
	const CParallelVecProjectionGeometry3D* parvec3dgeom = dynamic_cast<const CParallelVecProjectionGeometry3D*>(pProjGeom);
	const CConeVecProjectionGeometry3D* conevec3dgeom = dynamic_cast<const CConeVecProjectionGeometry3D*>(pProjGeom);

	if (conegeom || conevec3dgeom) {
		SConeProjection* pConeProjs;
		if (conegeom) {
			pConeProjs = getProjectionVectors<SConeProjection>(conegeom);
		} else {
			pConeProjs = getProjectionVectors<SConeProjection>(conevec3dgeom);
		}

		translateProjectionVectors(pConeProjs, pProjGeom->getProjectionCount(), v);

		CProjectionGeometry3D* ret = new CConeVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                              size,
		                                                              pProjGeom->getDetectorColCount(),
		                                                              pConeProjs);


		delete[] pConeProjs;
		return ret;
	} else {
		assert(par3dgeom || parvec3dgeom);
		SPar3DProjection* pParProjs;
		if (par3dgeom) {
			pParProjs = getProjectionVectors<SPar3DProjection>(par3dgeom);
		} else {
			pParProjs = getProjectionVectors<SPar3DProjection>(parvec3dgeom);
		}

		translateProjectionVectors(pParProjs, pProjGeom->getProjectionCount(), v);

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                  size,
		                                                                  pProjGeom->getDetectorColCount(),
		                                                                  pParProjs);

		delete[] pParProjs;
		return ret;
	}

}



// split self into sub-parts:
// - each no bigger than maxSize
// - number of sub-parts is divisible by div
// - maybe all approximately the same size?
CCompositeGeometryManager::TPartList CCompositeGeometryManager::CVolumePart::split(size_t maxSize, int div)
{
	TPartList ret;

	if (true) {
		// Split in vertical direction only at first, until we figure out
		// a model for splitting in other directions

		size_t sliceSize = ((size_t) pGeom->getGridColCount()) * pGeom->getGridRowCount();
		int sliceCount = pGeom->getGridSliceCount();
		size_t blockSize = computeVerticalSplit(maxSize / sliceSize, div, sliceCount);

		int rem = sliceCount % blockSize;

		ASTRA_DEBUG("From %d to %d step %d", -(rem / 2), sliceCount, blockSize);

		for (int z = -(rem / 2); z < sliceCount; z += blockSize) {
			int newsubZ = z;
			if (newsubZ < 0) newsubZ = 0;
			int endZ = z + blockSize;
			if (endZ > sliceCount) endZ = sliceCount;
			int size = endZ - newsubZ;

			CVolumePart *sub = new CVolumePart();
			sub->subX = this->subX;
			sub->subY = this->subY;
			sub->subZ = this->subZ + newsubZ;

			ASTRA_DEBUG("VolumePart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

			double shift = pGeom->getPixelLengthZ() * newsubZ;

			sub->pData = pData;
			sub->pGeom = new CVolumeGeometry3D(pGeom->getGridColCount(),
			                                   pGeom->getGridRowCount(),
			                                   size,
			                                   pGeom->getWindowMinX(),
			                                   pGeom->getWindowMinY(),
			                                   pGeom->getWindowMinZ() + shift,
			                                   pGeom->getWindowMaxX(),
			                                   pGeom->getWindowMaxY(),
			                                   pGeom->getWindowMinZ() + shift + size * pGeom->getPixelLengthZ());

			ret.push_back(boost::shared_ptr<CPart>(sub));
		}
	}

	return ret;
}

CCompositeGeometryManager::CVolumePart* CCompositeGeometryManager::CVolumePart::clone() const
{
	return new CVolumePart(*this);
}

CCompositeGeometryManager::CProjectionPart::CProjectionPart(const CProjectionPart& other)
 : CPart(other)
{
	pGeom = other.pGeom->clone();
}

CCompositeGeometryManager::CProjectionPart::~CProjectionPart()
{
	delete pGeom;
}

void CCompositeGeometryManager::CProjectionPart::getDims(size_t &x, size_t &y, size_t &z)
{
	if (!pGeom) {
		x = y = z = 0;
		return;
	}

	x = pGeom->getDetectorColCount();
	y = pGeom->getProjectionCount();
	z = pGeom->getDetectorRowCount();
}


CCompositeGeometryManager::CPart* CCompositeGeometryManager::CProjectionPart::reduce(const CPart *_other)
{
	const CVolumePart *other = dynamic_cast<const CVolumePart *>(_other);
	assert(other);

	double vmin_g, vmax_g;

	// reduce self to only cover intersection with projection of VolumePart
	// (Project corners of volume, take bounding box)

	for (int i = 0; i < pGeom->getProjectionCount(); ++i) {

		double vol_u[8];
		double vol_v[8];

		double pixx = other->pGeom->getPixelLengthX();
		double pixy = other->pGeom->getPixelLengthY();
		double pixz = other->pGeom->getPixelLengthZ();

		// TODO: Is 0.5 sufficient?
		double xmin = other->pGeom->getWindowMinX() - 0.5 * pixx;
		double xmax = other->pGeom->getWindowMaxX() + 0.5 * pixx;
		double ymin = other->pGeom->getWindowMinY() - 0.5 * pixy;
		double ymax = other->pGeom->getWindowMaxY() + 0.5 * pixy;
		double zmin = other->pGeom->getWindowMinZ() - 0.5 * pixz;
		double zmax = other->pGeom->getWindowMaxZ() + 0.5 * pixz;

		pGeom->projectPoint(xmin, ymin, zmin, i, vol_u[0], vol_v[0]);
		pGeom->projectPoint(xmin, ymin, zmax, i, vol_u[1], vol_v[1]);
		pGeom->projectPoint(xmin, ymax, zmin, i, vol_u[2], vol_v[2]);
		pGeom->projectPoint(xmin, ymax, zmax, i, vol_u[3], vol_v[3]);
		pGeom->projectPoint(xmax, ymin, zmin, i, vol_u[4], vol_v[4]);
		pGeom->projectPoint(xmax, ymin, zmax, i, vol_u[5], vol_v[5]);
		pGeom->projectPoint(xmax, ymax, zmin, i, vol_u[6], vol_v[6]);
		pGeom->projectPoint(xmax, ymax, zmax, i, vol_u[7], vol_v[7]);

		double vmin = vol_v[0];
		double vmax = vol_v[0];

		for (int j = 1; j < 8; ++j) {
			if (vol_v[j] < vmin)
				vmin = vol_v[j];
			if (vol_v[j] > vmax)
				vmax = vol_v[j];
		}

		if (i == 0 || vmin < vmin_g)
			vmin_g = vmin;
		if (i == 0 || vmax > vmax_g)
			vmax_g = vmax;
	}

	// fprintf(stderr, "v extent: %f %f\n", vmin_g, vmax_g);

	int _vmin = (int)floor(vmin_g - 1.0f);
	int _vmax = (int)ceil(vmax_g + 1.0f);
	if (_vmin < 0)
		_vmin = 0;
	if (_vmax > pGeom->getDetectorRowCount())
		_vmax = pGeom->getDetectorRowCount();

	if (_vmin >= _vmax) {
		_vmin = _vmax = 0;
	}

	CProjectionPart *sub = new CProjectionPart();
	sub->subX = this->subX;
	sub->subY = this->subY;
	sub->subZ = this->subZ + _vmin;

	sub->pData = pData;

	if (_vmin == _vmax) {
		sub->pGeom = 0;
	} else {
		sub->pGeom = getSubProjectionGeometry(pGeom, _vmin, _vmax - _vmin);
	}

	ASTRA_DEBUG("Reduce projection from %d - %d to %d - %d", this->subZ, this->subZ + pGeom->getDetectorRowCount(), this->subZ + _vmin, this->subZ + _vmax);

	return sub;
}


CCompositeGeometryManager::TPartList CCompositeGeometryManager::CProjectionPart::split(size_t maxSize, int div)
{
	TPartList ret;

	if (true) {
		// Split in vertical direction only at first, until we figure out
		// a model for splitting in other directions

		size_t sliceSize = ((size_t) pGeom->getDetectorColCount()) * pGeom->getProjectionCount();
		int sliceCount = pGeom->getDetectorRowCount();
		size_t blockSize = computeVerticalSplit(maxSize / sliceSize, div, sliceCount);

		int rem = sliceCount % blockSize;

		for (int z = -(rem / 2); z < sliceCount; z += blockSize) {
			int newsubZ = z;
			if (newsubZ < 0) newsubZ = 0;
			int endZ = z + blockSize;
			if (endZ > sliceCount) endZ = sliceCount;
			int size = endZ - newsubZ;

			CProjectionPart *sub = new CProjectionPart();
			sub->subX = this->subX;
			sub->subY = this->subY;
			sub->subZ = this->subZ + newsubZ;

			ASTRA_DEBUG("ProjectionPart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

			sub->pData = pData;

			sub->pGeom = getSubProjectionGeometry(pGeom, newsubZ, size);

			ret.push_back(boost::shared_ptr<CPart>(sub));
		}
	}

	return ret;

}

CCompositeGeometryManager::CProjectionPart* CCompositeGeometryManager::CProjectionPart::clone() const
{
	return new CProjectionPart(*this);
}


bool CCompositeGeometryManager::doFP(CProjector3D *pProjector, CFloat32VolumeData3DMemory *pVolData,
                                     CFloat32ProjectionData3DMemory *pProjData)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doFP");
	// Create single job for FP
	// Run result

	CVolumePart *input = new CVolumePart();
	input->pData = pVolData;
	input->subX = 0;
	input->subY = 0;
	input->subZ = 0;
	input->pGeom = pVolData->getGeometry()->clone();
	ASTRA_DEBUG("Main FP VolumePart -> %p", (void*)input);

	CProjectionPart *output = new CProjectionPart();
	output->pData = pProjData;
	output->subX = 0;
	output->subY = 0;
	output->subZ = 0;
	output->pGeom = pProjData->getGeometry()->clone();
	ASTRA_DEBUG("Main FP ProjectionPart -> %p", (void*)output);

	SJob FP;
	FP.pInput = boost::shared_ptr<CPart>(input);
	FP.pOutput = boost::shared_ptr<CPart>(output);
	FP.pProjector = pProjector;
	FP.eType = SJob::JOB_FP;
	FP.eMode = SJob::MODE_SET;

	TJobList L;
	L.push_back(FP);

	return doJobs(L);
}

bool CCompositeGeometryManager::doBP(CProjector3D *pProjector, CFloat32VolumeData3DMemory *pVolData,
                                     CFloat32ProjectionData3DMemory *pProjData)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doBP");
	// Create single job for BP
	// Run result

	CProjectionPart *input = new CProjectionPart();
	input->pData = pProjData;
	input->subX = 0;
	input->subY = 0;
	input->subZ = 0;
	input->pGeom = pProjData->getGeometry()->clone();

	CVolumePart *output = new CVolumePart();
	output->pData = pVolData;
	output->subX = 0;
	output->subY = 0;
	output->subZ = 0;
	output->pGeom = pVolData->getGeometry()->clone();

	SJob BP;
	BP.pInput = boost::shared_ptr<CPart>(input);
	BP.pOutput = boost::shared_ptr<CPart>(output);
	BP.pProjector = pProjector;
	BP.eType = SJob::JOB_BP;
	BP.eMode = SJob::MODE_SET;

	TJobList L;
	L.push_back(BP);

	return doJobs(L);
}

bool CCompositeGeometryManager::doFP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3DMemory *>& volData, const std::vector<CFloat32ProjectionData3DMemory *>& projData)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doFP, multi-volume");

	std::vector<CFloat32VolumeData3DMemory *>::const_iterator i;
	std::vector<boost::shared_ptr<CPart> > inputs;

	for (i = volData.begin(); i != volData.end(); ++i) {
		CVolumePart *input = new CVolumePart();
		input->pData = *i;
		input->subX = 0;
		input->subY = 0;
		input->subZ = 0;
		input->pGeom = (*i)->getGeometry()->clone();

		inputs.push_back(boost::shared_ptr<CPart>(input));
	}

	std::vector<CFloat32ProjectionData3DMemory *>::const_iterator j;
	std::vector<boost::shared_ptr<CPart> > outputs;

	for (j = projData.begin(); j != projData.end(); ++j) {
		CProjectionPart *output = new CProjectionPart();
		output->pData = *j;
		output->subX = 0;
		output->subY = 0;
		output->subZ = 0;
		output->pGeom = (*j)->getGeometry()->clone();

		outputs.push_back(boost::shared_ptr<CPart>(output));
	}

	std::vector<boost::shared_ptr<CPart> >::iterator i2;
	std::vector<boost::shared_ptr<CPart> >::iterator j2;
	TJobList L;

	for (i2 = outputs.begin(); i2 != outputs.end(); ++i2) {
		SJob FP;
		FP.eMode = SJob::MODE_SET;
		for (j2 = inputs.begin(); j2 != inputs.end(); ++j2) {
			FP.pInput = *j2;
			FP.pOutput = *i2;
			FP.pProjector = pProjector;
			FP.eType = SJob::JOB_FP;
			L.push_back(FP);

			// Set first, add rest
			FP.eMode = SJob::MODE_ADD;
		}
	}

	return doJobs(L);
}

bool CCompositeGeometryManager::doBP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3DMemory *>& volData, const std::vector<CFloat32ProjectionData3DMemory *>& projData)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doBP, multi-volume");


	std::vector<CFloat32VolumeData3DMemory *>::const_iterator i;
	std::vector<boost::shared_ptr<CPart> > outputs;

	for (i = volData.begin(); i != volData.end(); ++i) {
		CVolumePart *output = new CVolumePart();
		output->pData = *i;
		output->subX = 0;
		output->subY = 0;
		output->subZ = 0;
		output->pGeom = (*i)->getGeometry()->clone();

		outputs.push_back(boost::shared_ptr<CPart>(output));
	}

	std::vector<CFloat32ProjectionData3DMemory *>::const_iterator j;
	std::vector<boost::shared_ptr<CPart> > inputs;

	for (j = projData.begin(); j != projData.end(); ++j) {
		CProjectionPart *input = new CProjectionPart();
		input->pData = *j;
		input->subX = 0;
		input->subY = 0;
		input->subZ = 0;
		input->pGeom = (*j)->getGeometry()->clone();

		inputs.push_back(boost::shared_ptr<CPart>(input));
	}

	std::vector<boost::shared_ptr<CPart> >::iterator i2;
	std::vector<boost::shared_ptr<CPart> >::iterator j2;
	TJobList L;

	for (i2 = outputs.begin(); i2 != outputs.end(); ++i2) {
		SJob BP;
		BP.eMode = SJob::MODE_SET;
		for (j2 = inputs.begin(); j2 != inputs.end(); ++j2) {
			BP.pInput = *j2;
			BP.pOutput = *i2;
			BP.pProjector = pProjector;
			BP.eType = SJob::JOB_BP;
			L.push_back(BP);

			// Set first, add rest
			BP.eMode = SJob::MODE_ADD;
		}
	}

	return doJobs(L);
}




bool CCompositeGeometryManager::doJobs(TJobList &jobs)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doJobs");

	// Sort job list into job set by output part
	TJobSet jobset;

	for (TJobList::iterator i = jobs.begin(); i != jobs.end(); ++i) {
		jobset[i->pOutput.get()].push_back(*i);
	}

	size_t maxSize = astraCUDA3d::availableGPUMemory();
	if (maxSize == 0) {
		ASTRA_WARN("Unable to get available GPU memory. Defaulting to 1GB.");
		maxSize = 1024 * 1024 * 1024;
	} else {
		ASTRA_DEBUG("Detected %lu bytes of GPU memory", maxSize);
	}
	maxSize = (maxSize * 9) / 10;

	maxSize /= sizeof(float);
	int div = 1;

	// TODO: Multi-GPU support

	// Split jobs to fit
	TJobSet split;
	splitJobs(jobset, maxSize, div, split);
	jobset.clear();

	// Run jobs
	
	for (TJobSet::iterator iter = split.begin(); iter != split.end(); ++iter) {

		CPart* output = iter->first;
		TJobList& L = iter->second;

		assert(!L.empty());

		bool zero = L.begin()->eMode == SJob::MODE_SET;

		size_t outx, outy, outz;
		output->getDims(outx, outy, outz);

		if (L.begin()->eType == SJob::JOB_NOP) {
			// just zero output?
			if (zero) {
				for (size_t z = 0; z < outz; ++z) {
					for (size_t y = 0; y < outy; ++y) {
						float* ptr = output->pData->getData();
						ptr += (z + output->subX) * (size_t)output->pData->getHeight() * (size_t)output->pData->getWidth();
						ptr += (y + output->subY) * (size_t)output->pData->getWidth();
						ptr += output->subX;
						memset(ptr, 0, sizeof(float) * outx);
					}
				}
			}
			continue;
		}


		astraCUDA3d::SSubDimensions3D dstdims;
		dstdims.nx = output->pData->getWidth();
		dstdims.pitch = dstdims.nx;
		dstdims.ny = output->pData->getHeight();
		dstdims.nz = output->pData->getDepth();
		dstdims.subnx = outx;
		dstdims.subny = outy;
		dstdims.subnz = outz;
		ASTRA_DEBUG("dstdims: %d,%d,%d in %d,%d,%d", dstdims.subnx, dstdims.subny, dstdims.subnz, dstdims.nx, dstdims.ny, dstdims.nz);
		dstdims.subx = output->subX;
		dstdims.suby = output->subY;
		dstdims.subz = output->subZ;
		float *dst = output->pData->getData();

		astraCUDA3d::MemHandle3D outputMem = astraCUDA3d::allocateGPUMemory(outx, outy, outz, zero ? astraCUDA3d::INIT_ZERO : astraCUDA3d::INIT_NO);
		bool ok = outputMem;

		for (TJobList::iterator i = L.begin(); i != L.end(); ++i) {
			SJob &j = *i;

			assert(j.pInput);

			CCudaProjector3D *projector = dynamic_cast<CCudaProjector3D*>(j.pProjector);
			Cuda3DProjectionKernel projKernel = ker3d_default;
			int detectorSuperSampling = 1;
			int voxelSuperSampling = 1;
			if (projector) {
				projKernel = projector->getProjectionKernel();
				detectorSuperSampling = projector->getDetectorSuperSampling();
				voxelSuperSampling = projector->getVoxelSuperSampling();
			}

			size_t inx, iny, inz;
			j.pInput->getDims(inx, iny, inz);
			astraCUDA3d::MemHandle3D inputMem = astraCUDA3d::allocateGPUMemory(inx, iny, inz, astraCUDA3d::INIT_NO);

			astraCUDA3d::SSubDimensions3D srcdims;
			srcdims.nx = j.pInput->pData->getWidth();
			srcdims.pitch = srcdims.nx;
			srcdims.ny = j.pInput->pData->getHeight();
			srcdims.nz = j.pInput->pData->getDepth();
			srcdims.subnx = inx;
			srcdims.subny = iny;
			srcdims.subnz = inz;
			srcdims.subx = j.pInput->subX;
			srcdims.suby = j.pInput->subY;
			srcdims.subz = j.pInput->subZ;
			const float *src = j.pInput->pData->getDataConst();

			ok = astraCUDA3d::copyToGPUMemory(src, inputMem, srcdims);
			if (!ok) ASTRA_ERROR("Error copying input data to GPU");

			if (j.eType == SJob::JOB_FP) {
				assert(dynamic_cast<CVolumePart*>(j.pInput.get()));
				assert(dynamic_cast<CProjectionPart*>(j.pOutput.get()));

				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing FP");

				ok = astraCUDA3d::FP(((CProjectionPart*)j.pOutput.get())->pGeom, outputMem, ((CVolumePart*)j.pInput.get())->pGeom, inputMem, detectorSuperSampling, projKernel);
				if (!ok) ASTRA_ERROR("Error performing sub-FP");
				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: FP done");
			} else if (j.eType == SJob::JOB_BP) {
				assert(dynamic_cast<CVolumePart*>(j.pOutput.get()));
				assert(dynamic_cast<CProjectionPart*>(j.pInput.get()));

				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing BP");

				ok = astraCUDA3d::BP(((CProjectionPart*)j.pInput.get())->pGeom, inputMem, ((CVolumePart*)j.pOutput.get())->pGeom, outputMem, voxelSuperSampling);
				if (!ok) ASTRA_ERROR("Error performing sub-BP");
				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: BP done");
			} else {
				assert(false);
			}

			ok = astraCUDA3d::freeGPUMemory(inputMem);
			if (!ok) ASTRA_ERROR("Error freeing GPU memory");

		}

		ok = astraCUDA3d::copyFromGPUMemory(dst, outputMem, dstdims);
		if (!ok) ASTRA_ERROR("Error copying output data from GPU");
		
		ok = astraCUDA3d::freeGPUMemory(outputMem);
		if (!ok) ASTRA_ERROR("Error freeing GPU memory");
	}

	return true;
}



}

#endif
