/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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
#include "astra/Float32ProjectionData3DGPU.h"
#include "astra/Float32VolumeData3DGPU.h"
#include "astra/Logging.h"

#include "astra/cuda/2d/astra.h"
#include "astra/cuda/3d/mem3d.h"

#include <cstring>
#include <sstream>
#include <climits>

#ifndef USE_PTHREADS
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#endif


namespace astra {

SGPUParams* CCompositeGeometryManager::s_params = 0;

CCompositeGeometryManager::CCompositeGeometryManager()
{
	m_iMaxSize = 0;

	if (s_params) {
		m_iMaxSize = s_params->memory;
		m_GPUIndices = s_params->GPUIndices;
	}
}


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





class _AstraExport CFloat32CustomGPUMemory {
public:
    astraCUDA3d::MemHandle3D hnd; // Only required to be valid between allocate/free
    virtual bool allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, astraCUDA3d::Mem3DZeroMode zero)=0;
    virtual bool copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos)=0;
    virtual bool copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos)=0;
    virtual bool freeGPUMemory()=0;
	virtual ~CFloat32CustomGPUMemory() { }
};

class CFloat32ExistingGPUMemory : public astra::CFloat32CustomGPUMemory {
public:
    CFloat32ExistingGPUMemory(CFloat32Data3DGPU *d);
    virtual bool allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, astraCUDA3d::Mem3DZeroMode zero);
    virtual bool copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos);
    virtual bool copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos);
    virtual bool freeGPUMemory();

protected:
    unsigned int x, y, z;
};

class CFloat32DefaultGPUMemory : public astra::CFloat32CustomGPUMemory {
public:
	CFloat32DefaultGPUMemory(CFloat32Data3DMemory* d) {
		ptr = d->getData();
	}
	virtual bool allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, astraCUDA3d::Mem3DZeroMode zero) {
		hnd = astraCUDA3d::allocateGPUMemory(x, y, z, zero);
		return (bool)hnd;
	}
	virtual bool copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
		return astraCUDA3d::copyToGPUMemory(ptr, hnd, pos);
	}
	virtual bool copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
		return astraCUDA3d::copyFromGPUMemory(ptr, hnd, pos);
	}
	virtual bool freeGPUMemory() {
		return astraCUDA3d::freeGPUMemory(hnd);
	}

protected:
	float *ptr;
};



CFloat32ExistingGPUMemory::CFloat32ExistingGPUMemory(CFloat32Data3DGPU *d)
{
	hnd = d->getHandle();
	x = d->getWidth();
	y = d->getHeight();
	z = d->getDepth();
}

bool CFloat32ExistingGPUMemory::allocateGPUMemory(unsigned int x_, unsigned int y_, unsigned int z_, astraCUDA3d::Mem3DZeroMode zero) {
    assert(x_ == x);
    assert(y_ == y);
    assert(z_ == z);

    if (zero == astraCUDA3d::INIT_ZERO)
        return astraCUDA3d::zeroGPUMemory(hnd, x, y, z);
    else
        return true;
}
bool CFloat32ExistingGPUMemory::copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
    assert(pos.nx == x);
    assert(pos.ny == y);
    assert(pos.nz == z);
    assert(pos.pitch == x);
    assert(pos.subx == 0);
    assert(pos.suby == 0);
    assert(pos.subnx == x);
    assert(pos.subny == y);

    // These are less necessary than x/y, but allowing access to
    // subvolumes needs an interface change
    assert(pos.subz == 0);
    assert(pos.subnz == z);

    return true;
}
bool CFloat32ExistingGPUMemory::copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
    assert(pos.nx == x);
    assert(pos.ny == y);
    assert(pos.nz == z);
    assert(pos.pitch == x);
    assert(pos.subx == 0);
    assert(pos.suby == 0);
    assert(pos.subnx == x);
    assert(pos.subny == y);

    // These are less necessary than x/y, but allowing access to
    // subvolumes needs an interface change
    assert(pos.subz == 0);
    assert(pos.subnz == z);

    return true;
}
bool CFloat32ExistingGPUMemory::freeGPUMemory() {
    return true;
}


CFloat32CustomGPUMemory * createGPUMemoryHandler(CFloat32Data3D *d) {
	CFloat32Data3DMemory *dMem = dynamic_cast<CFloat32Data3DMemory*>(d);
	CFloat32Data3DGPU *dGPU = dynamic_cast<CFloat32Data3DGPU*>(d);

	if (dMem)
		return new CFloat32DefaultGPUMemory(dMem);
	else
		return new CFloat32ExistingGPUMemory(dGPU);
}





bool CCompositeGeometryManager::splitJobs(TJobSet &jobs, size_t maxSize, int div, TJobSet &split)
{
	int maxBlockDim = astraCUDA3d::maxBlockDimension();
	ASTRA_DEBUG("Found max block dim %d", maxBlockDim);

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

		TPartList splitOutput;
		pOutput->splitZ(splitOutput, maxSize/3, UINT_MAX, div);
#if 0
		TPartList splitOutput2;
		for (TPartList::iterator i_out = splitOutput.begin(); i_out != splitOutput.end(); ++i_out) {
			boost::shared_ptr<CPart> outputPart = *i_out;
			outputPart.get()->splitX(splitOutput2, UINT_MAX, UINT_MAX, 1);
		}
		splitOutput.clear();
		for (TPartList::iterator i_out = splitOutput2.begin(); i_out != splitOutput2.end(); ++i_out) {
			boost::shared_ptr<CPart> outputPart = *i_out;
					outputPart.get()->splitY(splitOutput, UINT_MAX, UINT_MAX, 1);
		}
		splitOutput2.clear();
#endif


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
				newjob.FDKSettings = j->FDKSettings;

				CPart* input = job.pInput->reduce(outputPart.get());

				if (input->getSize() == 0) {
					ASTRA_DEBUG("Empty input");
					newjob.eType = SJob::JOB_NOP;
					split[outputPart.get()].push_back(newjob);
					continue;
				}

				size_t remainingSize = ( maxSize - outputPart->getSize() ) / 2;

				TPartList splitInput;
				input->splitZ(splitInput, remainingSize, maxBlockDim, 1);
				delete input;
				TPartList splitInput2;
				for (TPartList::iterator i_in = splitInput.begin(); i_in != splitInput.end(); ++i_in) {
					boost::shared_ptr<CPart> inputPart = *i_in;
					inputPart.get()->splitX(splitInput2, UINT_MAX, maxBlockDim, 1);
				}
				splitInput.clear();
				for (TPartList::iterator i_in = splitInput2.begin(); i_in != splitInput2.end(); ++i_in) {
					boost::shared_ptr<CPart> inputPart = *i_in;
					inputPart.get()->splitY(splitInput, UINT_MAX, maxBlockDim, 1);
				}
				splitInput2.clear();

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


static std::pair<double, double> reduceProjectionVertical(const CVolumeGeometry3D* pVolGeom, const CProjectionGeometry3D* pProjGeom)
{
	double vmin_g, vmax_g;

	// reduce self to only cover intersection with projection of VolumePart
	// (Project corners of volume, take bounding box)

	assert(pProjGeom->getProjectionCount() > 0);
	for (int i = 0; i < pProjGeom->getProjectionCount(); ++i) {

		double vol_u[8];
		double vol_v[8];

		double pixx = pVolGeom->getPixelLengthX();
		double pixy = pVolGeom->getPixelLengthY();
		double pixz = pVolGeom->getPixelLengthZ();

		// TODO: Is 0.5 sufficient?
		double xmin = pVolGeom->getWindowMinX() - 0.5 * pixx;
		double xmax = pVolGeom->getWindowMaxX() + 0.5 * pixx;
		double ymin = pVolGeom->getWindowMinY() - 0.5 * pixy;
		double ymax = pVolGeom->getWindowMaxY() + 0.5 * pixy;
		double zmin = pVolGeom->getWindowMinZ() - 0.5 * pixz;
		double zmax = pVolGeom->getWindowMaxZ() + 0.5 * pixz;

		pProjGeom->projectPoint(xmin, ymin, zmin, i, vol_u[0], vol_v[0]);
		pProjGeom->projectPoint(xmin, ymin, zmax, i, vol_u[1], vol_v[1]);
		pProjGeom->projectPoint(xmin, ymax, zmin, i, vol_u[2], vol_v[2]);
		pProjGeom->projectPoint(xmin, ymax, zmax, i, vol_u[3], vol_v[3]);
		pProjGeom->projectPoint(xmax, ymin, zmin, i, vol_u[4], vol_v[4]);
		pProjGeom->projectPoint(xmax, ymin, zmax, i, vol_u[5], vol_v[5]);
		pProjGeom->projectPoint(xmax, ymax, zmin, i, vol_u[6], vol_v[6]);
		pProjGeom->projectPoint(xmax, ymax, zmax, i, vol_u[7], vol_v[7]);

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

	if (vmin_g < -1.0)
		vmin_g = -1.0;
	if (vmax_g > pProjGeom->getDetectorRowCount())
		vmax_g = pProjGeom->getDetectorRowCount();

	if (vmin_g >= vmax_g)
		vmin_g = vmax_g = 0.0;

	return std::pair<double, double>(vmin_g, vmax_g);

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

void CCompositeGeometryManager::CVolumePart::getDims(size_t &x, size_t &y, size_t &z) const
{
	if (!pGeom) {
		x = y = z = 0;
		return;
	}

	x = pGeom->getGridColCount();
	y = pGeom->getGridRowCount();
	z = pGeom->getGridSliceCount();
}

size_t CCompositeGeometryManager::CPart::getSize() const
{
	size_t x, y, z;
	getDims(x, y, z);
	return x * y * z;
}

bool CCompositeGeometryManager::CPart::isFull() const
{
	size_t x, y, z;
	getDims(x, y, z);
	return x == (size_t)pData->getWidth() &&
	       y == (size_t)pData->getHeight() &&
	       z == (size_t)pData->getDepth();
}

bool CCompositeGeometryManager::CPart::canSplitAndReduce() const
{
	return dynamic_cast<CFloat32Data3DMemory *>(pData) != 0;
}



static bool testVolumeRange(const std::pair<double, double>& fullRange,
                            const CVolumeGeometry3D *pVolGeom,
							const CProjectionGeometry3D *pProjGeom,
							int zmin, int zmax)
{
	double pixz = pVolGeom->getPixelLengthZ();

	CVolumeGeometry3D test(pVolGeom->getGridColCount(),
	                       pVolGeom->getGridRowCount(),
	                       zmax - zmin,
	                       pVolGeom->getWindowMinX(),
	                       pVolGeom->getWindowMinY(),
	                       pVolGeom->getWindowMinZ() + zmin * pixz,
	                       pVolGeom->getWindowMaxX(),
	                       pVolGeom->getWindowMaxY(),
	                       pVolGeom->getWindowMinZ() + zmax * pixz);


	std::pair<double, double> subRange = reduceProjectionVertical(&test, pProjGeom);

	// empty
	if (subRange.first == subRange.second)
		return true;

	// fully outside of fullRange
	if (subRange.first >= fullRange.second || subRange.second <= fullRange.first)
		return true;

	return false;
}


CCompositeGeometryManager::CPart* CCompositeGeometryManager::CVolumePart::reduce(const CPart *_other)
{
	if (!canSplitAndReduce())
		return clone();

	const CProjectionPart *other = dynamic_cast<const CProjectionPart *>(_other);
	assert(other);


	std::pair<double, double> fullRange = reduceProjectionVertical(pGeom, other->pGeom);

	int top_slice = 0, bottom_slice = 0;

	if (fullRange.first < fullRange.second) {


		// TOP SLICE

		int zmin = 0;
		int zmax = pGeom->getGridSliceCount()-1; // (Don't try empty region)

		// Setting top slice to zmin is always valid.

		while (zmin < zmax) {
			int zmid = (zmin + zmax + 1) / 2;

			bool ok = testVolumeRange(fullRange, pGeom, other->pGeom,
			                          0, zmid);

			ASTRA_DEBUG("binsearch min: [%d,%d], %d, %s", zmin, zmax, zmid, ok ? "ok" : "removed too much");

			if (ok)
				zmin = zmid;
			else
				zmax = zmid - 1;
		}

		top_slice = zmin;


		// BOTTOM SLICE

		zmin = top_slice + 1; // (Don't try empty region)
		zmax = pGeom->getGridSliceCount();

		// Setting bottom slice to zmax is always valid

		while (zmin < zmax) {
			int zmid = (zmin + zmax) / 2;

			bool ok = testVolumeRange(fullRange, pGeom, other->pGeom,
			                          zmid, pGeom->getGridSliceCount());

			ASTRA_DEBUG("binsearch max: [%d,%d], %d, %s", zmin, zmax, zmid, ok ? "ok" : "removed too much");

			if (ok)
				zmax = zmid;
			else
				zmin = zmid + 1;

		}

		bottom_slice = zmax;

	}

	ASTRA_DEBUG("found extent: %d - %d", top_slice, bottom_slice);

	top_slice -= 1;
	if (top_slice < 0)
		top_slice = 0;
	bottom_slice += 1;
	if (bottom_slice >= pGeom->getGridSliceCount())
		bottom_slice = pGeom->getGridSliceCount();

	ASTRA_DEBUG("adjusted extent: %d - %d", top_slice, bottom_slice);

	double pixz = pGeom->getPixelLengthZ();

	CVolumePart *sub = new CVolumePart();
	sub->subX = this->subX;
	sub->subY = this->subY;
	sub->subZ = this->subZ + top_slice;
	sub->pData = pData;

	if (top_slice == bottom_slice) {
		sub->pGeom = 0;
	} else {
		sub->pGeom = new CVolumeGeometry3D(pGeom->getGridColCount(),
		                                   pGeom->getGridRowCount(),
		                                   bottom_slice - top_slice,
		                                   pGeom->getWindowMinX(),
		                                   pGeom->getWindowMinY(),
		                                   pGeom->getWindowMinZ() + top_slice * pixz,
		                                   pGeom->getWindowMaxX(),
		                                   pGeom->getWindowMaxY(),
		                                   pGeom->getWindowMinZ() + bottom_slice * pixz);
	}

	ASTRA_DEBUG("Reduce volume from %d - %d to %d - %d ( %f - %f )", this->subZ, this->subZ +  pGeom->getGridSliceCount(), this->subZ + top_slice, this->subZ + bottom_slice, pGeom->getWindowMinZ() + top_slice * pixz, pGeom->getWindowMinZ() + bottom_slice * pixz);

	return sub;
}



static size_t ceildiv(size_t a, size_t b) {
	return (a + b - 1) / b;
}

static size_t computeLinearSplit(size_t maxBlock, int div, size_t sliceCount)
{
	size_t blockSize = maxBlock;
	size_t blockCount;
	if (sliceCount <= blockSize)
		blockCount = 1;
	else
		blockCount = ceildiv(sliceCount, blockSize);

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
static void translateProjectionVectorsU(V* pProjs, int count, double du)
{
	for (int i = 0; i < count; ++i) {
		pProjs[i].fDetSX += du * pProjs[i].fDetUX;
		pProjs[i].fDetSY += du * pProjs[i].fDetUY;
		pProjs[i].fDetSZ += du * pProjs[i].fDetUZ;
	}
}

template<class V>
static void translateProjectionVectorsV(V* pProjs, int count, double dv)
{
	for (int i = 0; i < count; ++i) {
		pProjs[i].fDetSX += dv * pProjs[i].fDetVX;
		pProjs[i].fDetSY += dv * pProjs[i].fDetVY;
		pProjs[i].fDetSZ += dv * pProjs[i].fDetVZ;
	}
}


static CProjectionGeometry3D* getSubProjectionGeometryU(const CProjectionGeometry3D* pProjGeom, int u, int size)
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

		translateProjectionVectorsU(pConeProjs, pProjGeom->getProjectionCount(), u);

		CProjectionGeometry3D* ret = new CConeVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                              pProjGeom->getDetectorRowCount(),
		                                                              size,
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

		translateProjectionVectorsU(pParProjs, pProjGeom->getProjectionCount(), u);

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                  pProjGeom->getDetectorRowCount(),
		                                                                  size,
		                                                                  pParProjs);

		delete[] pParProjs;
		return ret;
	}

}



static CProjectionGeometry3D* getSubProjectionGeometryV(const CProjectionGeometry3D* pProjGeom, int v, int size)
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

		translateProjectionVectorsV(pConeProjs, pProjGeom->getProjectionCount(), v);

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

		translateProjectionVectorsV(pParProjs, pProjGeom->getProjectionCount(), v);

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(pProjGeom->getProjectionCount(),
		                                                                  size,
		                                                                  pProjGeom->getDetectorColCount(),
		                                                                  pParProjs);

		delete[] pParProjs;
		return ret;
	}

}

static CProjectionGeometry3D* getSubProjectionGeometryAngle(const CProjectionGeometry3D* pProjGeom, int th, int size)
{
	// First convert to vectors, then convert into new object

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

		CProjectionGeometry3D* ret = new CConeVecProjectionGeometry3D(size,
		                                                              pProjGeom->getDetectorRowCount(),
		                                                              pProjGeom->getDetectorColCount(),
		                                                              pConeProjs + th);


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

		CProjectionGeometry3D* ret = new CParallelVecProjectionGeometry3D(size,
		                                                                  pProjGeom->getDetectorRowCount(),
		                                                                  pProjGeom->getDetectorColCount(),
		                                                                  pParProjs + th);

		delete[] pParProjs;
		return ret;
	}

}



// split self into sub-parts:
// - each no bigger than maxSize
// - number of sub-parts is divisible by div
// - maybe all approximately the same size?
void CCompositeGeometryManager::CVolumePart::splitX(CCompositeGeometryManager::TPartList& out, size_t maxSize, size_t maxDim, int div)
{
	if (canSplitAndReduce()) {
		size_t sliceSize = ((size_t) pGeom->getGridSliceCount()) * pGeom->getGridRowCount();
		int sliceCount = pGeom->getGridColCount();
		size_t m = std::min(maxSize / sliceSize, maxDim);
		size_t blockSize = computeLinearSplit(m, div, sliceCount);

		int rem = blockSize - (sliceCount % blockSize);
		if ((size_t)rem == blockSize)
			rem = 0;

		ASTRA_DEBUG("From %d to %d step %d", -(rem / 2), sliceCount, blockSize);

		for (int x = -(rem / 2); x < sliceCount; x += blockSize) {
			int newsubX = x;
			if (newsubX < 0) newsubX = 0;
			int endX = x + blockSize;
			if (endX > sliceCount) endX = sliceCount;
			int size = endX - newsubX;

			CVolumePart *sub = new CVolumePart();
			sub->subX = this->subX + newsubX;
			sub->subY = this->subY;
			sub->subZ = this->subZ;

			ASTRA_DEBUG("VolumePart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

			double shift = pGeom->getPixelLengthX() * newsubX;

			sub->pData = pData;
			sub->pGeom = new CVolumeGeometry3D(size,
			                                   pGeom->getGridRowCount(),
			                                   pGeom->getGridSliceCount(),
			                                   pGeom->getWindowMinX() + shift,
			                                   pGeom->getWindowMinY(),
			                                   pGeom->getWindowMinZ(),
			                                   pGeom->getWindowMinX() + shift + size * pGeom->getPixelLengthX(),
			                                   pGeom->getWindowMaxY(),
			                                   pGeom->getWindowMaxZ());

			out.push_back(boost::shared_ptr<CPart>(sub));
		}
	} else {
		out.push_back(boost::shared_ptr<CPart>(clone()));
	}
}

void CCompositeGeometryManager::CVolumePart::splitY(CCompositeGeometryManager::TPartList& out, size_t maxSize, size_t maxDim, int div)
{
	if (canSplitAndReduce()) {
		size_t sliceSize = ((size_t) pGeom->getGridColCount()) * pGeom->getGridSliceCount();
		int sliceCount = pGeom->getGridRowCount();
		size_t m = std::min(maxSize / sliceSize, maxDim);
		size_t blockSize = computeLinearSplit(m, div, sliceCount);

		int rem = blockSize - (sliceCount % blockSize);
		if ((size_t)rem == blockSize)
			rem = 0;

		ASTRA_DEBUG("From %d to %d step %d", -(rem / 2), sliceCount, blockSize);

		for (int y = -(rem / 2); y < sliceCount; y += blockSize) {
			int newsubY = y;
			if (newsubY < 0) newsubY = 0;
			int endY = y + blockSize;
			if (endY > sliceCount) endY = sliceCount;
			int size = endY - newsubY;

			CVolumePart *sub = new CVolumePart();
			sub->subX = this->subX;
			sub->subY = this->subY + newsubY;
			sub->subZ = this->subZ;

			ASTRA_DEBUG("VolumePart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

			double shift = pGeom->getPixelLengthY() * newsubY;

			sub->pData = pData;
			sub->pGeom = new CVolumeGeometry3D(pGeom->getGridColCount(),
			                                   size,
			                                   pGeom->getGridSliceCount(),
			                                   pGeom->getWindowMinX(),
			                                   pGeom->getWindowMinY() + shift,
			                                   pGeom->getWindowMinZ(),
			                                   pGeom->getWindowMaxX(),
			                                   pGeom->getWindowMinY() + shift + size * pGeom->getPixelLengthY(),
			                                   pGeom->getWindowMaxZ());

			out.push_back(boost::shared_ptr<CPart>(sub));
		}
	} else {
		out.push_back(boost::shared_ptr<CPart>(clone()));
	}
}

void CCompositeGeometryManager::CVolumePart::splitZ(CCompositeGeometryManager::TPartList& out, size_t maxSize, size_t maxDim, int div)
{
	if (canSplitAndReduce()) {
		size_t sliceSize = ((size_t) pGeom->getGridColCount()) * pGeom->getGridRowCount();
		int sliceCount = pGeom->getGridSliceCount();
		size_t m = std::min(maxSize / sliceSize, maxDim);
		size_t blockSize = computeLinearSplit(m, div, sliceCount);

		int rem = blockSize - (sliceCount % blockSize);
		if ((size_t)rem == blockSize)
			rem = 0;

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

			out.push_back(boost::shared_ptr<CPart>(sub));
		}
	} else {
		out.push_back(boost::shared_ptr<CPart>(clone()));
	}
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

void CCompositeGeometryManager::CProjectionPart::getDims(size_t &x, size_t &y, size_t &z) const
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
	if (!canSplitAndReduce())
		return clone();

	const CVolumePart *other = dynamic_cast<const CVolumePart *>(_other);
	assert(other);

	std::pair<double, double> r = reduceProjectionVertical(other->pGeom, pGeom);
	// fprintf(stderr, "v extent: %f %f\n", r.first, r.second);
	int _vmin = (int)floor(r.first - 1.0);
	int _vmax = (int)ceil(r.second + 1.0);
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
		sub->pGeom = getSubProjectionGeometryV(pGeom, _vmin, _vmax - _vmin);
	}

	ASTRA_DEBUG("Reduce projection from %d - %d to %d - %d", this->subZ, this->subZ + pGeom->getDetectorRowCount(), this->subZ + _vmin, this->subZ + _vmax);

	return sub;
}


void CCompositeGeometryManager::CProjectionPart::splitX(CCompositeGeometryManager::TPartList &out, size_t maxSize, size_t maxDim, int div)
{
	if (canSplitAndReduce()) {
		size_t sliceSize = ((size_t) pGeom->getDetectorRowCount()) * pGeom->getProjectionCount();
		int sliceCount = pGeom->getDetectorColCount();
		size_t m = std::min(maxSize / sliceSize, maxDim);
		size_t blockSize = computeLinearSplit(m, div, sliceCount);

		int rem = blockSize - (sliceCount % blockSize);
		if ((size_t)rem == blockSize)
			rem = 0;

		ASTRA_DEBUG("From %d to %d step %d", -(rem / 2), sliceCount, blockSize);

		for (int x = -(rem / 2); x < sliceCount; x += blockSize) {
			int newsubX = x;
			if (newsubX < 0) newsubX = 0;
			int endX = x + blockSize;
			if (endX > sliceCount) endX = sliceCount;
			int size = endX - newsubX;

			CProjectionPart *sub = new CProjectionPart();
			sub->subX = this->subX + newsubX;
			sub->subY = this->subY;
			sub->subZ = this->subZ;

			ASTRA_DEBUG("ProjectionPart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

			sub->pData = pData;

			sub->pGeom = getSubProjectionGeometryU(pGeom, newsubX, size);

			out.push_back(boost::shared_ptr<CPart>(sub));
		}
	} else {
		out.push_back(boost::shared_ptr<CPart>(clone()));
	}
}

void CCompositeGeometryManager::CProjectionPart::splitY(CCompositeGeometryManager::TPartList &out, size_t maxSize, size_t maxDim, int div)
{
	if (canSplitAndReduce()) {
		size_t sliceSize = ((size_t) pGeom->getDetectorColCount()) * pGeom->getDetectorRowCount();
		int angleCount = pGeom->getProjectionCount();
		size_t m = std::min(maxSize / sliceSize, maxDim);
		size_t blockSize = computeLinearSplit(m, div, angleCount);

		ASTRA_DEBUG("From %d to %d step %d", 0, angleCount, blockSize);

		for (int th = 0; th < angleCount; th += blockSize) {
			int endTh = th + blockSize;
			if (endTh > angleCount) endTh = angleCount;
			int size = endTh - th;

			CProjectionPart *sub = new CProjectionPart();
			sub->subX = this->subX;
			sub->subY = this->subY + th;
			sub->subZ = this->subZ;

			ASTRA_DEBUG("ProjectionPart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

			sub->pData = pData;

			sub->pGeom = getSubProjectionGeometryAngle(pGeom, th, size);

			out.push_back(boost::shared_ptr<CPart>(sub));
		}
	} else {
		out.push_back(boost::shared_ptr<CPart>(clone()));
	}
}

void CCompositeGeometryManager::CProjectionPart::splitZ(CCompositeGeometryManager::TPartList &out, size_t maxSize, size_t maxDim, int div)
{
	if (canSplitAndReduce()) {
		size_t sliceSize = ((size_t) pGeom->getDetectorColCount()) * pGeom->getProjectionCount();
		int sliceCount = pGeom->getDetectorRowCount();
		size_t m = std::min(maxSize / sliceSize, maxDim);
		size_t blockSize = computeLinearSplit(m, div, sliceCount);

		int rem = blockSize - (sliceCount % blockSize);
		if ((size_t)rem == blockSize)
			rem = 0;

		ASTRA_DEBUG("From %d to %d step %d", -(rem / 2), sliceCount, blockSize);

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

			sub->pGeom = getSubProjectionGeometryV(pGeom, newsubZ, size);

			out.push_back(boost::shared_ptr<CPart>(sub));
		}
	} else {
		out.push_back(boost::shared_ptr<CPart>(clone()));
	}

}

CCompositeGeometryManager::CProjectionPart* CCompositeGeometryManager::CProjectionPart::clone() const
{
	return new CProjectionPart(*this);
}

CCompositeGeometryManager::SJob CCompositeGeometryManager::createJobFP(CProjector3D *pProjector,
                                            CFloat32VolumeData3D *pVolData,
                                            CFloat32ProjectionData3D *pProjData,
                                            SJob::EMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::createJobFP");
	// Create single job for FP

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
	FP.eMode = eMode;

	return FP;
}

CCompositeGeometryManager::SJob CCompositeGeometryManager::createJobBP(CProjector3D *pProjector,
                                            CFloat32VolumeData3D *pVolData,
                                            CFloat32ProjectionData3D *pProjData,
                                            SJob::EMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::createJobBP");
	// Create single job for BP

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
	BP.eMode = eMode;

	return BP;
}

bool CCompositeGeometryManager::doFP(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
                                     CFloat32ProjectionData3D *pProjData, SJob::EMode eMode)
{
	TJobList L;
	L.push_back(createJobFP(pProjector, pVolData, pProjData, eMode));

	return doJobs(L);
}

		bool CCompositeGeometryManager::doBP(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
                                     CFloat32ProjectionData3D *pProjData, SJob::EMode eMode)
{
	TJobList L;
	L.push_back(createJobBP(pProjector, pVolData, pProjData, eMode));

	return doJobs(L);
}


bool CCompositeGeometryManager::doFDK(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
                                     CFloat32ProjectionData3D *pProjData, bool bShortScan,
                                     const float *pfFilter, SJob::EMode eMode)
{
	if (!dynamic_cast<CConeProjectionGeometry3D*>(pProjData->getGeometry()) &&
	    !dynamic_cast<CConeVecProjectionGeometry3D*>(pProjData->getGeometry())) {
		ASTRA_ERROR("CCompositeGeometryManager::doFDK: cone/cone_vec geometry required");
		return false;
	}

	SJob job = createJobBP(pProjector, pVolData, pProjData, eMode);
	job.eType = SJob::JOB_FDK;
	job.FDKSettings.bShortScan = bShortScan;
	job.FDKSettings.pfFilter = pfFilter;

	TJobList L;
	L.push_back(job);

	return doJobs(L);
}

bool CCompositeGeometryManager::doFP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3D *>& volData, const std::vector<CFloat32ProjectionData3D *>& projData, SJob::EMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doFP, multi-volume");

	std::vector<CFloat32VolumeData3D *>::const_iterator i;
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

	std::vector<CFloat32ProjectionData3D *>::const_iterator j;
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
		FP.eMode = eMode;
		for (j2 = inputs.begin(); j2 != inputs.end(); ++j2) {
			FP.pInput = *j2;
			FP.pOutput = *i2;
			FP.pProjector = pProjector;
			FP.eType = SJob::JOB_FP;
			L.push_back(FP);

			// Always ADD rest
			FP.eMode = SJob::MODE_ADD;
		}
	}

	return doJobs(L);
}

bool CCompositeGeometryManager::doBP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3D *>& volData, const std::vector<CFloat32ProjectionData3D *>& projData, SJob::EMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doBP, multi-volume");


	std::vector<CFloat32VolumeData3D *>::const_iterator i;
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

	std::vector<CFloat32ProjectionData3D *>::const_iterator j;
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
		BP.eMode = eMode;
		for (j2 = inputs.begin(); j2 != inputs.end(); ++j2) {
			BP.pInput = *j2;
			BP.pOutput = *i2;
			BP.pProjector = pProjector;
			BP.eType = SJob::JOB_BP;
			L.push_back(BP);

			// Always ADD rest
			BP.eMode = SJob::MODE_ADD;
		}
	}

	return doJobs(L);
}




static bool doJob(const CCompositeGeometryManager::TJobSet::const_iterator& iter)
{
	CCompositeGeometryManager::CPart* output = iter->first;
	const CCompositeGeometryManager::TJobList& L = iter->second;

	assert(!L.empty());

	bool zero = L.begin()->eMode == CCompositeGeometryManager::SJob::MODE_SET;

	size_t outx, outy, outz;
	output->getDims(outx, outy, outz);

	if (L.begin()->eType == CCompositeGeometryManager::SJob::JOB_NOP) {
		// just zero output?
		if (zero) {
			// TODO: This function shouldn't have to know about this difference
			// between Memory/GPU
			CFloat32Data3DMemory *hostMem = dynamic_cast<CFloat32Data3DMemory *>(output->pData);
			if (hostMem) {
				for (size_t z = 0; z < outz; ++z) {
					for (size_t y = 0; y < outy; ++y) {
						float* ptr = hostMem->getData();
						ptr += (z + output->subX) * (size_t)output->pData->getHeight() * (size_t)output->pData->getWidth();
						ptr += (y + output->subY) * (size_t)output->pData->getWidth();
						ptr += output->subX;
						memset(ptr, 0, sizeof(float) * outx);
					}
				}
			} else {
				CFloat32Data3DGPU *gpuMem = dynamic_cast<CFloat32Data3DGPU *>(output->pData);
				assert(gpuMem);
				assert(output->isFull()); // TODO: zero subset?

				zeroGPUMemory(gpuMem->getHandle(), outx, outy, outz);
			}
		}
		return true;
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

	CFloat32CustomGPUMemory *dstMem = createGPUMemoryHandler(output->pData);

	bool ok = dstMem->allocateGPUMemory(outx, outy, outz, zero ? astraCUDA3d::INIT_ZERO : astraCUDA3d::INIT_NO);
	if (!ok) ASTRA_ERROR("Error allocating GPU memory");

	if (!zero) {
		// instead of zeroing output memory, copy from host
		ok = dstMem->copyToGPUMemory(dstdims);
		if (!ok) ASTRA_ERROR("Error copying output data to GPU");
	}

	for (CCompositeGeometryManager::TJobList::const_iterator i = L.begin(); i != L.end(); ++i) {
		const CCompositeGeometryManager::SJob &j = *i;

		assert(j.pInput);

		CCudaProjector3D *projector = dynamic_cast<CCudaProjector3D*>(j.pProjector);
		Cuda3DProjectionKernel projKernel = ker3d_default;
		int detectorSuperSampling = 1;
		int voxelSuperSampling = 1;
		bool densityWeighting = false;
		if (projector) {
			projKernel = projector->getProjectionKernel();
			detectorSuperSampling = projector->getDetectorSuperSampling();
			voxelSuperSampling = projector->getVoxelSuperSampling();
			densityWeighting = projector->getDensityWeighting();
		}

		size_t inx, iny, inz;
		j.pInput->getDims(inx, iny, inz);

		CFloat32CustomGPUMemory *srcMem = createGPUMemoryHandler(j.pInput->pData);

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

		ok = srcMem->allocateGPUMemory(inx, iny, inz, astraCUDA3d::INIT_NO);
		if (!ok) ASTRA_ERROR("Error allocating GPU memory");

		ok = srcMem->copyToGPUMemory(srcdims);
		if (!ok) ASTRA_ERROR("Error copying input data to GPU");

		switch (j.eType) {
		case CCompositeGeometryManager::SJob::JOB_FP:
		{
			assert(dynamic_cast<CCompositeGeometryManager::CVolumePart*>(j.pInput.get()));
			assert(dynamic_cast<CCompositeGeometryManager::CProjectionPart*>(j.pOutput.get()));

			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing FP");

			ok = astraCUDA3d::FP(((CCompositeGeometryManager::CProjectionPart*)j.pOutput.get())->pGeom, dstMem->hnd, ((CCompositeGeometryManager::CVolumePart*)j.pInput.get())->pGeom, srcMem->hnd, detectorSuperSampling, projKernel);
			if (!ok) ASTRA_ERROR("Error performing sub-FP");
			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: FP done");
		}
		break;
		case CCompositeGeometryManager::SJob::JOB_BP:
		{
			assert(dynamic_cast<CCompositeGeometryManager::CVolumePart*>(j.pOutput.get()));
			assert(dynamic_cast<CCompositeGeometryManager::CProjectionPart*>(j.pInput.get()));

			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing BP");

			ok = astraCUDA3d::BP(((CCompositeGeometryManager::CProjectionPart*)j.pInput.get())->pGeom, srcMem->hnd, ((CCompositeGeometryManager::CVolumePart*)j.pOutput.get())->pGeom, dstMem->hnd, voxelSuperSampling, densityWeighting);
			if (!ok) ASTRA_ERROR("Error performing sub-BP");
			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: BP done");
		}
		break;
		case CCompositeGeometryManager::SJob::JOB_FDK:
		{
			assert(dynamic_cast<CCompositeGeometryManager::CVolumePart*>(j.pOutput.get()));
			assert(dynamic_cast<CCompositeGeometryManager::CProjectionPart*>(j.pInput.get()));

			if (srcdims.subx || srcdims.suby) {
				ASTRA_ERROR("CCompositeGeometryManager::doJobs: data too large for FDK");
				ok = false;
			} else {
				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing FDK");

				ok = astraCUDA3d::FDK(((CCompositeGeometryManager::CProjectionPart*)j.pInput.get())->pGeom, srcMem->hnd, ((CCompositeGeometryManager::CVolumePart*)j.pOutput.get())->pGeom, dstMem->hnd, j.FDKSettings.bShortScan, j.FDKSettings.pfFilter);
				if (!ok) ASTRA_ERROR("Error performing sub-FDK");
				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: FDK done");
			}
		}
		break;
		default:
			assert(false);
		}

		ok = srcMem->freeGPUMemory();
		if (!ok) ASTRA_ERROR("Error freeing GPU memory");

		delete srcMem;
	}

	ok = dstMem->copyFromGPUMemory(dstdims);
	if (!ok) ASTRA_ERROR("Error copying output data from GPU");
	
	ok = dstMem->freeGPUMemory();
	if (!ok) ASTRA_ERROR("Error freeing GPU memory");

	delete dstMem;

	return true;
}


class WorkQueue {
public:
	WorkQueue(CCompositeGeometryManager::TJobSet &_jobs) : m_jobs(_jobs) {
#ifdef USE_PTHREADS
		pthread_mutex_init(&m_mutex, 0);
#endif
		m_iter = m_jobs.begin();
	}
	bool receive(CCompositeGeometryManager::TJobSet::const_iterator &i) {
		lock();

		if (m_iter == m_jobs.end()) {
			unlock();
			return false;
		}

		i = m_iter++;

		unlock();

		return true;	
	}
#ifdef USE_PTHREADS
	void lock() {
		// TODO: check mutex op return values
		pthread_mutex_lock(&m_mutex);
	}
	void unlock() {
		// TODO: check mutex op return values
		pthread_mutex_unlock(&m_mutex);
	}
#else
	void lock() {
		m_mutex.lock();
	}
	void unlock() {
		m_mutex.unlock();
	}
#endif

private:
	CCompositeGeometryManager::TJobSet &m_jobs;
	CCompositeGeometryManager::TJobSet::const_iterator m_iter;
#ifdef USE_PTHREADS
	pthread_mutex_t m_mutex;
#else
	boost::mutex m_mutex;
#endif
};

struct WorkThreadInfo {
	WorkQueue* m_queue;
	unsigned int m_iGPU;
};

#ifndef USE_PTHREADS

void runEntries_boost(WorkThreadInfo* info)
{
	ASTRA_DEBUG("Launching thread on GPU %d\n", info->m_iGPU);
	CCompositeGeometryManager::TJobSet::const_iterator i;
	while (info->m_queue->receive(i)) {
		ASTRA_DEBUG("Running block on GPU %d\n", info->m_iGPU);
		astraCUDA3d::setGPUIndex(info->m_iGPU);
		boost::this_thread::interruption_point();
		doJob(i);
		boost::this_thread::interruption_point();
	}
	ASTRA_DEBUG("Finishing thread on GPU %d\n", info->m_iGPU);
}


#else

void* runEntries_pthreads(void* data) {
	WorkThreadInfo* info = (WorkThreadInfo*)data;

	ASTRA_DEBUG("Launching thread on GPU %d\n", info->m_iGPU);

	CCompositeGeometryManager::TJobSet::const_iterator i;

	while (info->m_queue->receive(i)) {
		ASTRA_DEBUG("Running block on GPU %d\n", info->m_iGPU);
		astraCUDA3d::setGPUIndex(info->m_iGPU);
		pthread_testcancel();
		doJob(i);
		pthread_testcancel();
	}
	ASTRA_DEBUG("Finishing thread on GPU %d\n", info->m_iGPU);

	return 0;
}

#endif


void runWorkQueue(WorkQueue &queue, const std::vector<int> & iGPUIndices) {
	int iThreadCount = iGPUIndices.size();

	std::vector<WorkThreadInfo> infos;
#ifdef USE_PTHREADS
	std::vector<pthread_t> threads;
#else
	std::vector<boost::thread*> threads;
#endif
	infos.resize(iThreadCount);
	threads.resize(iThreadCount);

	for (int i = 0; i < iThreadCount; ++i) {
		infos[i].m_queue = &queue;
		infos[i].m_iGPU = iGPUIndices[i];
#ifdef USE_PTHREADS
		pthread_create(&threads[i], 0, runEntries_pthreads, (void*)&infos[i]);
#else
		threads[i] = new boost::thread(runEntries_boost, &infos[i]);
#endif
	}

	// Wait for them to finish
	for (int i = 0; i < iThreadCount; ++i) {
#ifdef USE_PTHREADS
		pthread_join(threads[i], 0);
#else
		threads[i]->join();
		delete threads[i];
		threads[i] = 0;
#endif
	}
}


void CCompositeGeometryManager::setGPUIndices(const std::vector<int>& GPUIndices)
{
	m_GPUIndices = GPUIndices;
}

bool CCompositeGeometryManager::doJobs(TJobList &jobs)
{
	// TODO: Proper clean up if substeps fail (Or as proper as possible)

	ASTRA_DEBUG("CCompositeGeometryManager::doJobs");

	// Sort job list into job set by output part
	TJobSet jobset;

	for (TJobList::iterator i = jobs.begin(); i != jobs.end(); ++i) {
		jobset[i->pOutput.get()].push_back(*i);
	}

	size_t maxSize = m_iMaxSize;
	if (maxSize == 0) {
		// Get memory from first GPU. Not optimal...
		if (!m_GPUIndices.empty())
			astraCUDA3d::setGPUIndex(m_GPUIndices[0]);
		maxSize = astraCUDA::availableGPUMemory();
		if (maxSize == 0) {
			ASTRA_WARN("Unable to get available GPU memory. Defaulting to 1GB.");
			maxSize = 1024 * 1024 * 1024;
		} else {
			ASTRA_DEBUG("Detected %lu bytes of GPU memory", maxSize);
		}
	} else {
		ASTRA_DEBUG("Set to %lu bytes of GPU memory", maxSize);
	}
	maxSize = (maxSize * 9) / 10;

	maxSize /= sizeof(float);
	int div = 1;
	if (!m_GPUIndices.empty())
		div = m_GPUIndices.size();

	// Split jobs to fit
	TJobSet split;
	splitJobs(jobset, maxSize, div, split);
	jobset.clear();

	if (m_GPUIndices.size() <= 1) {

		// Run jobs
		ASTRA_DEBUG("Running single-threaded");

		if (!m_GPUIndices.empty())
			astraCUDA3d::setGPUIndex(m_GPUIndices[0]);

		for (TJobSet::const_iterator iter = split.begin(); iter != split.end(); ++iter) {
			doJob(iter);
		}

	} else {

		ASTRA_DEBUG("Running multi-threaded");

		WorkQueue wq(split);

		runWorkQueue(wq, m_GPUIndices);

	}

	return true;
}




//static
void CCompositeGeometryManager::setGlobalGPUParams(const SGPUParams& params)
{
	delete s_params;

	s_params = new SGPUParams;
	*s_params = params;

	ASTRA_DEBUG("CompositeGeometryManager: Setting global GPU params:");
	std::ostringstream s;
	s << "GPU indices:";
	for (unsigned int i = 0; i < params.GPUIndices.size(); ++i)
		s << " " << params.GPUIndices[i];
	std::string ss = s.str();
	ASTRA_DEBUG(ss.c_str());
	ASTRA_DEBUG("Memory: %llu", params.memory);
}


}

#endif
