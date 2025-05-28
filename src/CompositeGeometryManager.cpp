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

#include "astra/CompositeGeometryManager.h"

#ifdef ASTRA_CUDA

#include "astra/GeometryUtil3D.h"
#include "astra/VolumeGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/CylConeVecProjectionGeometry3D.h"
#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/Projector3D.h"
#include "astra/CudaProjector3D.h"
#include "astra/Data3D.h"
#include "astra/Logging.h"

#include "astra/cuda/2d/astra.h"
#include "astra/cuda/3d/mem3d.h"

#include <cstring>
#include <sstream>
#include <climits>
#include <mutex>
#include <thread>
#include <atomic>


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



class _AstraExport CGPUMemory {
public:
	astraCUDA3d::MemHandle3D hnd; // Only required to be valid between allocate/free
	virtual bool allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, astraCUDA3d::Mem3DZeroMode zero)=0;
	virtual bool copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos)=0;
	virtual bool copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos)=0;
	virtual ~CGPUMemory() { }
};

class CExistingGPUMemory : public astra::CGPUMemory {
public:
	CExistingGPUMemory(CData3D *d);
	virtual bool allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, astraCUDA3d::Mem3DZeroMode zero);
	virtual bool copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos);
	virtual bool copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos);
private:
	unsigned int x, y, z;
};

class CDefaultGPUMemory : public astra::CGPUMemory {
public:
	CDefaultGPUMemory(CData3D* d) : ptr(nullptr) {
		assert(d->getStorage()->isMemory());
		if (d->getStorage()->isFloat32())
			ptr = dynamic_cast<CDataMemory<float32>*>(d->getStorage())->getData();
		else
			assert(false);
	}
	virtual ~CDefaultGPUMemory() {
		freeGPUMemory();
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
private:
	float *ptr;
	void freeGPUMemory() {
		astraCUDA3d::freeGPUMemory(hnd);
	}
};



CExistingGPUMemory::CExistingGPUMemory(CData3D *d)
{
	assert(d->getStorage()->isGPU());
	CDataGPU *storage = dynamic_cast<CDataGPU*>(d->getStorage());

	hnd = storage->getHandle();
	x = d->getWidth();
	y = d->getHeight();
	z = d->getDepth();
}

bool CExistingGPUMemory::allocateGPUMemory(unsigned int x_, unsigned int y_, unsigned int z_, astraCUDA3d::Mem3DZeroMode zero) {
	assert(x_ == x);
	assert(y_ == y);
	assert(z_ == z);

	if (zero == astraCUDA3d::INIT_ZERO)
		return astraCUDA3d::zeroGPUMemory(hnd, x, y, z);
	else
		return true;
}
bool CExistingGPUMemory::copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
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
bool CExistingGPUMemory::copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
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


CGPUMemory * createGPUMemoryHandler(CData3D *d) {
	if (d->getStorage()->isMemory())
		return new CDefaultGPUMemory(d);
	else
		return new CExistingGPUMemory(d);
}



static astraCUDA3d::SSubDimensions3D getPartSubDims(const CCompositeGeometryManager::CPart *part)
{
	size_t subnx, subny, subnz;
	part->getDims(subnx, subny, subnz);

	astraCUDA3d::SSubDimensions3D d;
	d.nx = part->pData->getWidth();
	d.pitch = d.nx;
	d.ny = part->pData->getHeight();
	d.nz = part->pData->getDepth();
	d.subnx = subnx;
	d.subny = subny;
	d.subnz = subnz;
	d.subx = part->subX;
	d.suby = part->subY;
	d.subz = part->subZ;

	return d;
}

static void splitPart(int n,
                      std::unique_ptr<CCompositeGeometryManager::CPart> && base,
                      CCompositeGeometryManager::TPartList& out,
                      size_t maxSize, size_t maxDim, int div);

static std::unique_ptr<CCompositeGeometryManager::CPart>
reducePart(const CCompositeGeometryManager::CPart *base,
           const CCompositeGeometryManager::CPart *other);


bool CCompositeGeometryManager::splitJobs(TJobSetInternal &jobs, size_t maxSize, int div, TJobSetInternal &split)
{
	int maxBlockDim = astraCUDA3d::maxBlockDimension();
	ASTRA_DEBUG("Found max block dim %d", maxBlockDim);

	split.clear();

	size_t costHeuristic = 0;

	for (TJobSetInternal::iterator i = jobs.begin(); i != jobs.end(); ++i)
	{
		std::unique_ptr<CPart> pOutput = std::move(i->first);
		TJobListInternal &L = i->second;

		// 1. Split output part
		// 2. Per sub-part:
		//    a. reduce input part
		//    b. split input part
		//    c. create jobs for new (input,output) subparts

		TPartList splitOutput;
		// We now split projection data over the angle axis by default,
		// and volume data over the z axis.
		int axisOutput = 2;
		if (pOutput->eType == CPart::PART_PROJ)
			axisOutput = 1;

		splitPart(axisOutput, std::move(pOutput), splitOutput, maxSize/3, UINT_MAX, div);
#if 0
		// There are currently no reasons to split the output over other axes

		TPartList splitOutput2;
		for (std::unique_ptr<CPart> &i_out : splitOutput) {
			splitPart(axisOutputSecond, std::move(i_out), splitOutput2, UINT_MAX, UINT_MAX, 1);
		}
		splitOutput.clear();
		for (std::unique_ptr<CPart> &i_out : splitOutput2) {
			splitPart(axisOutputThird, std::move(i_out), splitOutput, UINT_MAX, UINT_MAX, 1);
		}
		splitOutput2.clear();
#endif

		for (TPartList::iterator i_out = splitOutput.begin();
		     i_out != splitOutput.end(); ++i_out)
		{
			TJobListInternal newjobs;

			std::unique_ptr<CPart> outputPart{std::move(*i_out)};
			for (const SJobInternal &job : L)
			{
				std::unique_ptr<CPart> input = reducePart(job.pInput.get(), outputPart.get());


				if (input->getSize() == 0) {
					ASTRA_DEBUG("Empty input");
					SJobInternal newjob{nullptr};
					newjob.eMode = job.eMode;
					newjob.pProjector = job.pProjector;
					newjob.FDKSettings = job.FDKSettings;
					newjob.eType = JOB_NOP;
					newjobs.push_back(std::move(newjob));
					continue;
				}

				size_t remainingSize = ( maxSize - outputPart->getSize() ) / 2;

				int axisInputFirst = 2;
				int axisInputSecond = 0;
				int axisInputThird = 1;
				if (input->eType == CPart::PART_PROJ) {
					axisInputFirst = 1;
					axisInputSecond = 2;
					axisInputThird = 0;
				}

				// We do two passes: first split along all dimensions only on maxBlockDim,
				// and then split again along the first axis on memory size.
				TPartList splitInput;
				splitPart(axisInputFirst, std::move(input), splitInput, 1024ULL*1024*1024*1024, maxBlockDim, 1);
				TPartList splitInput2;
				for (std::unique_ptr<CPart> &inputPart : splitInput)
					splitPart(axisInputSecond, std::move(inputPart), splitInput2, 1024ULL*1024*1024*1024, maxBlockDim, 1);

				splitInput.clear();

				TPartList splitInput3;
				for (std::unique_ptr<CPart> &inputPart : splitInput2)
					splitPart(axisInputThird, std::move(inputPart), splitInput3, 1024ULL*1024*1024*1024, maxBlockDim, 1);

				splitInput2.clear();
				for (std::unique_ptr<CPart> &inputPart : splitInput3)
					splitPart(axisInputFirst, std::move(inputPart), splitInput, remainingSize, maxBlockDim, 1);
				splitInput3.clear();

				ASTRA_DEBUG("Input split into %zu parts", splitInput.size());

				EJobMode eMode = job.eMode;

				for (std::unique_ptr<CPart> &i_in : splitInput) {
					SJobInternal newjob{std::move(i_in)};
					newjob.eMode = eMode;
					newjob.pProjector = job.pProjector;
					newjob.FDKSettings = job.FDKSettings;
					newjob.eType = job.eType;

					size_t tx, ty, tz;
					newjob.pInput->getDims(tx, ty, tz);

					switch (newjob.eType) {
						case JOB_FP:
							costHeuristic += outputPart.get()->getSize() * cbrt(tx * ty * tz);
							break;
						case JOB_BP: case JOB_FDK:
							costHeuristic += outputPart.get()->getSize() * ty;
							break;
						case JOB_NOP:
							break;
					}

					newjobs.push_back(std::move(newjob));

					// Second and later (input) parts should always be added to
					// output of first (input) part.
					eMode = MODE_ADD;
				}

			}
			split.push_back(std::make_pair(std::move(outputPart), std::move(newjobs)));
		}
	}

	ASTRA_DEBUG("splitJobs cost heuristic: %zu", costHeuristic);

	return true;
}


static std::pair<double, double> reduceProjectionVertical(const CVolumeGeometry3D* pVolGeom, const CProjectionGeometry3D* pProjGeom)
{
	double umin_g, umax_g;
	double vmin_g, vmax_g;

	double pixx = pVolGeom->getPixelLengthX();
	double pixy = pVolGeom->getPixelLengthY();
	double pixz = pVolGeom->getPixelLengthZ();

	pProjGeom->getProjectedBBox(pVolGeom->getWindowMinX() - 0.5 * pixx,
	                            pVolGeom->getWindowMaxX() + 0.5 * pixx,
	                            pVolGeom->getWindowMinY() - 0.5 * pixy,
	                            pVolGeom->getWindowMaxY() + 0.5 * pixy,
	                            pVolGeom->getWindowMinZ() - 0.5 * pixz,
	                            pVolGeom->getWindowMaxZ() + 0.5 * pixz,
	                            umin_g, umax_g,
	                            vmin_g, vmax_g);

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

CCompositeGeometryManager::CVolumePart::CVolumePart(CFloat32VolumeData3D *pVolData)
{
	eType = PART_VOL;
	pData = pVolData;
	pGeom = pVolData->getGeometry().clone();
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
	return pData->getStorage()->isMemory();
}


static CCompositeGeometryManager::CVolumePart* createSubVolumePart(const CCompositeGeometryManager::CVolumePart* base,
                                                                   unsigned int offset_x,
                                                                   unsigned int offset_y,
                                                                   unsigned int offset_z,
                                                                   CVolumeGeometry3D *pGeom)
{
	CCompositeGeometryManager::CVolumePart *sub = new CCompositeGeometryManager::CVolumePart();
	sub->subX = base->subX + offset_x;
	sub->subY = base->subY + offset_y;
	sub->subZ = base->subZ + offset_z;

	sub->pData = base->pData;

	sub->pGeom = pGeom;

	return sub;
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


static std::unique_ptr<CCompositeGeometryManager::CPart>
reduceVolumePart(const CCompositeGeometryManager::CVolumePart *base,
                     const CCompositeGeometryManager::CPart *_other )
{
	if (!base->canSplitAndReduce())
		return std::unique_ptr<CCompositeGeometryManager::CVolumePart>(base->clone());

	const CCompositeGeometryManager::CProjectionPart *other = dynamic_cast<const CCompositeGeometryManager::CProjectionPart *>(_other);
	assert(other);


	std::pair<double, double> fullRange = reduceProjectionVertical(base->pGeom, other->pGeom);

	int top_slice = 0, bottom_slice = 0;

	if (fullRange.first < fullRange.second) {


		// TOP SLICE

		int zmin = 0;
		int zmax = base->pGeom->getGridSliceCount()-1; // (Don't try empty region)

		// Setting top slice to zmin is always valid.

		while (zmin < zmax) {
			int zmid = (zmin + zmax + 1) / 2;

			// Test if this top area is entirely out of range
			bool ok = testVolumeRange(fullRange, base->pGeom, other->pGeom,
			                          0, zmid);

			//ASTRA_DEBUG("binsearch min: [%d,%d], %d, %s", zmin, zmax, zmid, ok ? "ok" : "removed too much");

			if (ok)
				zmin = zmid;
			else
				zmax = zmid - 1;
		}

		top_slice = zmin;


		// BOTTOM SLICE

		zmin = top_slice + 1; // (Don't try empty region)
		zmax = base->pGeom->getGridSliceCount();

		// Setting bottom slice to zmax is always valid

		while (zmin < zmax) {
			int zmid = (zmin + zmax) / 2;

			// Test if this bottom area is entirely out of range
			bool ok = testVolumeRange(fullRange, base->pGeom, other->pGeom,
			                          zmid, base->pGeom->getGridSliceCount());

			//ASTRA_DEBUG("binsearch max: [%d,%d], %d, %s", zmin, zmax, zmid, ok ? "ok" : "removed too much");

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
	if (bottom_slice >= base->pGeom->getGridSliceCount())
		bottom_slice = base->pGeom->getGridSliceCount();

	ASTRA_DEBUG("adjusted extent: %d - %d", top_slice, bottom_slice);

	double pixz = base->pGeom->getPixelLengthZ();

	CVolumeGeometry3D *pSubGeom = 0;
	if (top_slice != bottom_slice) {
		pSubGeom = new CVolumeGeometry3D(base->pGeom->getGridColCount(),
		                                 base->pGeom->getGridRowCount(),
		                                 bottom_slice - top_slice,
		                                 base->pGeom->getWindowMinX(),
		                                 base->pGeom->getWindowMinY(),
		                                 base->pGeom->getWindowMinZ() + top_slice * pixz,
		                                 base->pGeom->getWindowMaxX(),
		                                 base->pGeom->getWindowMaxY(),
		                                 base->pGeom->getWindowMinZ() + bottom_slice * pixz);
	}

	CCompositeGeometryManager::CVolumePart *sub = createSubVolumePart(base, 0, 0, top_slice, pSubGeom);

	ASTRA_DEBUG("Reduce volume from %d - %d to %d - %d ( %f - %f )", base->subZ, base->subZ + base->pGeom->getGridSliceCount(), base->subZ + top_slice, base->subZ + bottom_slice, base->pGeom->getWindowMinZ() + top_slice * pixz, base->pGeom->getWindowMinZ() + bottom_slice * pixz);

	return std::unique_ptr<CCompositeGeometryManager::CPart>(sub);
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

// split self into sub-parts:
// - each no bigger than maxSize
// - number of sub-parts is divisible by div
// - maybe all approximately the same size?

static void splitVolumePart(int n,
                            const CCompositeGeometryManager::CPart *base,
                            CCompositeGeometryManager::TPartList& out,
                            size_t maxSize, size_t maxDim, int div)
{
	assert(base->canSplitAndReduce());
	assert(n >= 0 && n < 3);

	const CCompositeGeometryManager::CVolumePart *pPart;
	pPart = dynamic_cast<const CCompositeGeometryManager::CVolumePart*>(base);
	assert(pPart);

	const CVolumeGeometry3D *pGeom = pPart->pGeom;

	size_t dims[3];
	pPart->getDims(dims[0], dims[1], dims[2]);
	size_t sliceSize = 1;
	for (int i = 0; i < 3; ++i)
		if (i != n)
			sliceSize *= dims[i];
	int sliceCount = dims[n];

	size_t m = std::min(maxSize / sliceSize, maxDim);
	size_t blockSize = computeLinearSplit(m, div, sliceCount);

	int rem = blockSize - (sliceCount % blockSize);
	if ((size_t)rem == blockSize)
		rem = 0;

	ASTRA_DEBUG("From %d to %d step %zu", -(rem / 2), sliceCount, blockSize);

	for (int x = -(rem / 2); x < sliceCount; x += blockSize) {
		int newsubX = x;
		if (newsubX < 0) newsubX = 0;
		int endX = x + blockSize;
		if (endX > sliceCount) endX = sliceCount;
		int size = endX - newsubX;

		int offsets[3] = { 0, 0, 0 };
		offsets[n] = newsubX;
		size_t sizes[3] = { dims[0], dims[1], dims[2] };
		sizes[n] = size;

		double shifts[3] = { 0., 0., 0. };
		if (n == 0)
			shifts[0] = pGeom->getPixelLengthX() * newsubX;
		else if (n == 1)
			shifts[1] = pGeom->getPixelLengthY() * newsubX;
		else if (n == 2)
			shifts[2] = pGeom->getPixelLengthZ() * newsubX;

		CVolumeGeometry3D *pSubGeom = new CVolumeGeometry3D(sizes[0],
		                                                    sizes[1],
		                                                    sizes[2],
		                                                    pGeom->getWindowMinX() + shifts[0],
		                                                    pGeom->getWindowMinY() + shifts[1],
		                                                    pGeom->getWindowMinZ() + shifts[2],
		                                                    pGeom->getWindowMinX() + shifts[0] + sizes[0] * pGeom->getPixelLengthX(),
		                                                    pGeom->getWindowMinY() + shifts[1] + sizes[1] * pGeom->getPixelLengthY(),
		                                                    pGeom->getWindowMinZ() + shifts[2] + sizes[2] * pGeom->getPixelLengthZ()
		                                                   );
		CCompositeGeometryManager::CVolumePart *sub = createSubVolumePart(pPart, offsets[0], offsets[1], offsets[2], pSubGeom);
		ASTRA_DEBUG("VolumePart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);

		out.push_back(std::unique_ptr<CCompositeGeometryManager::CPart>(sub));
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

CCompositeGeometryManager::CProjectionPart::CProjectionPart(CFloat32ProjectionData3D *pProjData)
{
	eType = PART_PROJ;
	pData = pProjData;
	pGeom = pProjData->getGeometry().clone();
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

static std::pair<int, int> reduceProjectionAngular(const CVolumeGeometry3D* pVolGeom, const CProjectionGeometry3D* pProjGeom)
{
	int iFirstAngle = pProjGeom->getProjectionCount();
	int iLastAngle = -1;
	for (int i = 0; i < pProjGeom->getProjectionCount(); ++i) {
		double umin, umax;
		double vmin, vmax;

		double pixx = pVolGeom->getPixelLengthX();
		double pixy = pVolGeom->getPixelLengthY();
		double pixz = pVolGeom->getPixelLengthZ();

		pProjGeom->getProjectedBBoxSingleAngle(i,
		                            pVolGeom->getWindowMinX() - 0.5 * pixx,
		                            pVolGeom->getWindowMaxX() + 0.5 * pixx,
		                            pVolGeom->getWindowMinY() - 0.5 * pixy,
		                            pVolGeom->getWindowMaxY() + 0.5 * pixy,
		                            pVolGeom->getWindowMinZ() - 0.5 * pixz,
		                            pVolGeom->getWindowMaxZ() + 0.5 * pixz,
		                            umin, umax,
		                            vmin, vmax);

		bool out = umin >= pProjGeom->getDetectorColCount() || umax <= 0 ||
		           vmin >= pProjGeom->getDetectorRowCount() || vmax <= 0;

		if (!out && i <= iFirstAngle)
			iFirstAngle = i;
		if (!out)
			iLastAngle = i;
	}

	ASTRA_DEBUG("reduceProjectionAngular: found [%d,%d]", iFirstAngle, iLastAngle);

	return std::pair<int, int>(iFirstAngle, iLastAngle);
}

static CCompositeGeometryManager::CProjectionPart* createSubProjectionPart(const CCompositeGeometryManager::CProjectionPart* base,
                                                                           unsigned int offset_u,
                                                                           unsigned int offset_th,
                                                                           unsigned int offset_v,
                                                                           CProjectionGeometry3D *pGeom)
{
	CCompositeGeometryManager::CProjectionPart *sub = new CCompositeGeometryManager::CProjectionPart();
	sub->subX = base->subX + offset_u;
	sub->subY = base->subY + offset_th;
	sub->subZ = base->subZ + offset_v;

	sub->pData = base->pData;

	sub->pGeom = pGeom;

	return sub;
}

static std::unique_ptr<CCompositeGeometryManager::CPart>
reduceProjectionPart(const CCompositeGeometryManager::CProjectionPart *base,
                     const CCompositeGeometryManager::CPart *_other)
{
	if (!base->canSplitAndReduce())
		return std::unique_ptr<CCompositeGeometryManager::CProjectionPart>(base->clone());

	const CCompositeGeometryManager::CVolumePart *other = dynamic_cast<const CCompositeGeometryManager::CVolumePart *>(_other);
	assert(other);

	std::pair<int, int> angleRange = reduceProjectionAngular(other->pGeom, base->pGeom);
	if (angleRange.first > angleRange.second) {
		ASTRA_DEBUG("Reduce projection: no angular overlap");
		return std::unique_ptr<CCompositeGeometryManager::CPart>(createSubProjectionPart(base, 0, 0, 0, nullptr));
	}
	ASTRA_DEBUG("Reduce projection: angular from %d - %d to %d - %d", base->subY, base->subY + base->pGeom->getProjectionCount(), base->subY + angleRange.first, base->subY + angleRange.second);

	CProjectionGeometry3D *pSubGeom1 = getSubProjectionGeometry_Angle(base->pGeom, angleRange.first, angleRange.second - angleRange.first + 1);

	// sub->subY += angleRange.first;


	std::pair<double, double> r = reduceProjectionVertical(other->pGeom, pSubGeom1);

	// fprintf(stderr, "v extent: %f %f\n", r.first, r.second);
	int _vmin = (int)floor(r.first - 1.0);
	int _vmax = (int)ceil(r.second + 1.0);
	if (_vmin < 0)
		_vmin = 0;
	if (_vmax > pSubGeom1->getDetectorRowCount())
		_vmax = pSubGeom1->getDetectorRowCount();

	if (_vmin >= _vmax) {
		_vmin = _vmax = 0;
	}

	CProjectionGeometry3D *pSubGeom2 = 0;
	if (_vmin != _vmax)
		pSubGeom2 = getSubProjectionGeometry_V(pSubGeom1, _vmin, _vmax - _vmin);
	CCompositeGeometryManager::CProjectionPart *sub = createSubProjectionPart(base, 0, 0, _vmin, pSubGeom2);

	delete pSubGeom1;

	ASTRA_DEBUG("Reduce projection: rows from %d - %d to %d - %d", base->subZ, base->subZ + base->pGeom->getDetectorRowCount(), base->subZ + _vmin, base->subZ + _vmax);

	return std::unique_ptr<CCompositeGeometryManager::CPart>(sub);
}

static void splitProjectionPart(int n,
                                const CCompositeGeometryManager::CPart *base,
                                CCompositeGeometryManager::TPartList &out,
                                size_t maxSize, size_t maxDim, int div)
{
	assert(base->canSplitAndReduce());
	assert(n >= 0 && n < 3);

	const CCompositeGeometryManager::CProjectionPart *pPart;
	pPart = dynamic_cast<const CCompositeGeometryManager::CProjectionPart*>(base);
	assert(pPart);

	const CProjectionGeometry3D *pGeom = pPart->pGeom;

	size_t dims[3];
	pPart->getDims(dims[0], dims[1], dims[2]);
	size_t sliceSize = 1;
	for (int i = 0; i < 3; ++i)
		if (i != n)
			sliceSize *= dims[i];
	int sliceCount = dims[n];
	size_t m = std::min(maxSize / sliceSize, maxDim);
	size_t blockSize = computeLinearSplit(m, div, sliceCount);

	int rem = blockSize - (sliceCount % blockSize);
	if ((size_t)rem == blockSize)
		rem = 0;

	// When splitting angles, just start at 0
	if (n == 1)
		rem = 0;

	ASTRA_DEBUG("From %d to %d step %zu", -(rem / 2), sliceCount, blockSize);


	for (int x = -(rem / 2); x < sliceCount; x += blockSize) {
		int newsubX = x;
		if (newsubX < 0) newsubX = 0;
		int endX = x + blockSize;
		if (endX > sliceCount) endX = sliceCount;
		int size = endX - newsubX;

		int offsets[3] = { 0, 0, 0 };
		offsets[n] = newsubX;

		CProjectionGeometry3D *pSubGeom = 0;
		if (n == 0)
			pSubGeom = getSubProjectionGeometry_U(pGeom, newsubX, size);
		else if (n == 1)
			pSubGeom = getSubProjectionGeometry_Angle(pGeom, newsubX, size);
		else if (n == 2)
			pSubGeom = getSubProjectionGeometry_V(pGeom, newsubX, size);

		CCompositeGeometryManager::CProjectionPart *sub;
		sub = createSubProjectionPart(pPart, offsets[0], offsets[1], offsets[2], pSubGeom);
		ASTRA_DEBUG("ProjectionPart split %d %d %d -> %p", sub->subX, sub->subY, sub->subZ, (void*)sub);
		out.push_back(std::unique_ptr<CCompositeGeometryManager::CPart>(sub));
	}
}

CCompositeGeometryManager::CProjectionPart* CCompositeGeometryManager::CProjectionPart::clone() const
{
	return new CProjectionPart(*this);
}

static void splitPart(int n,
                      std::unique_ptr<CCompositeGeometryManager::CPart> && base,
                      CCompositeGeometryManager::TPartList& out,
                      size_t maxSize, size_t maxDim, int div)
{
	if (!base.get()->canSplitAndReduce()) {
		out.push_back(std::move(base));
		return;
	}

	if (base.get()->eType == CCompositeGeometryManager::CPart::PART_PROJ)
		splitProjectionPart(n, base.get(), out, maxSize, maxDim, div);
	else if (base.get()->eType == CCompositeGeometryManager::CPart::PART_VOL)
		splitVolumePart(n, base.get(), out, maxSize, maxDim, div);
	else
		assert(false);
}

static std::unique_ptr<CCompositeGeometryManager::CPart>
reducePart(const CCompositeGeometryManager::CPart *base,
           const CCompositeGeometryManager::CPart *other )
{
	if (base->eType == CCompositeGeometryManager::CPart::PART_PROJ)
		return reduceProjectionPart(dynamic_cast<const CCompositeGeometryManager::CProjectionPart*>(base), other);
	else if (base->eType == CCompositeGeometryManager::CPart::PART_VOL)
		return reduceVolumePart(dynamic_cast<const CCompositeGeometryManager::CVolumePart*>(base), other);
	else
		assert(false);
}


CCompositeGeometryManager::SJob CCompositeGeometryManager::createJobFP(CProjector3D *pProjector,
                                            CFloat32VolumeData3D *pVolData,
                                            CFloat32ProjectionData3D *pProjData,
                                            EJobMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::createJobFP");
	// Create single job for FP

	CVolumePart *input = new CVolumePart{pVolData};
	ASTRA_DEBUG("Main FP VolumePart -> %p", (void*)input);

	CProjectionPart *output = new CProjectionPart{pProjData};
	ASTRA_DEBUG("Main FP ProjectionPart -> %p", (void*)output);

	SJob FP{std::unique_ptr<CPart>(input), std::unique_ptr<CPart>(output)};
	FP.pProjector = pProjector;
	FP.eType = JOB_FP;
	FP.eMode = eMode;

	return FP;
}

CCompositeGeometryManager::SJob CCompositeGeometryManager::createJobBP(CProjector3D *pProjector,
                                            CFloat32VolumeData3D *pVolData,
                                            CFloat32ProjectionData3D *pProjData,
                                            EJobMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::createJobBP");
	// Create single job for BP

	CProjectionPart *input = new CProjectionPart{pProjData};

	CVolumePart *output = new CVolumePart{pVolData};

	SJob BP{std::unique_ptr<CPart>(input), std::unique_ptr<CPart>(output)};
	BP.pProjector = pProjector;
	BP.eType = JOB_BP;
	BP.eMode = eMode;

	return BP;
}

bool CCompositeGeometryManager::doFP(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
                                     CFloat32ProjectionData3D *pProjData, EJobMode eMode)
{
	TJobList L;
	L.push_back(createJobFP(pProjector, pVolData, pProjData, eMode));

	return doJobs(L);
}

bool CCompositeGeometryManager::doBP(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
                                     CFloat32ProjectionData3D *pProjData, EJobMode eMode)
{
	TJobList L;
	L.push_back(createJobBP(pProjector, pVolData, pProjData, eMode));

	return doJobs(L);
}


bool CCompositeGeometryManager::doFDK(CProjector3D *pProjector, CFloat32VolumeData3D *pVolData,
                                     CFloat32ProjectionData3D *pProjData, bool bShortScan,
                                     const SFilterConfig &filterConfig, EJobMode eMode)
{
	if (!dynamic_cast<const CConeProjectionGeometry3D*>(&pProjData->getGeometry()) &&
	    !dynamic_cast<const CConeVecProjectionGeometry3D*>(&pProjData->getGeometry())) {
		ASTRA_ERROR("CCompositeGeometryManager::doFDK: cone/cone_vec geometry required");
		return false;
	}

	SJob job{createJobBP(pProjector, pVolData, pProjData, eMode)};
	job.eType = JOB_FDK;
	job.FDKSettings.bShortScan = bShortScan;
	job.FDKSettings.filterConfig = filterConfig;

	TJobList L;
	L.push_back(std::move(job));

	return doJobs(L);
}

bool CCompositeGeometryManager::doFP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3D *>& volData, const std::vector<CFloat32ProjectionData3D *>& projData, EJobMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doFP, multi-volume");

	TJobSetInternal jobset;

	for (CFloat32ProjectionData3D *j : projData) {
		CProjectionPart *output = new CProjectionPart{j};
		TJobListInternal L;
		EJobMode eNewMode = eMode;
		for (CFloat32VolumeData3D *i : volData) {
			CVolumePart *input = new CVolumePart{i};
			SJobInternal FP{std::unique_ptr<CPart>(input)};
			FP.eMode = eNewMode;
			FP.pProjector = pProjector;
			FP.eType = JOB_FP;
			L.push_back(std::move(FP));

			// Always ADD rest
			eNewMode = MODE_ADD;
		}
		jobset.push_back(std::make_pair(std::unique_ptr<CPart>(output), std::move(L)));
	}

	return doJobs(jobset);
}

bool CCompositeGeometryManager::doBP(CProjector3D *pProjector, const std::vector<CFloat32VolumeData3D *>& volData, const std::vector<CFloat32ProjectionData3D *>& projData, EJobMode eMode)
{
	ASTRA_DEBUG("CCompositeGeometryManager::doBP, multi-volume");

	TJobSetInternal jobset;

	for (CFloat32VolumeData3D *i : volData) {
		CVolumePart *output = new CVolumePart{i};
		TJobListInternal L;
		EJobMode eNewMode = eMode;
		for (CFloat32ProjectionData3D *j : projData) {
			CProjectionPart *input = new CProjectionPart{j};
			SJobInternal BP{std::unique_ptr<CPart>(input)};
			BP.eMode = eNewMode;
			BP.pProjector = pProjector;
			BP.eType = JOB_BP;
			L.push_back(std::move(BP));

			// Always ADD rest
			eNewMode = MODE_ADD;
		}
		jobset.push_back(std::make_pair(std::unique_ptr<CPart>(output), std::move(L)));
	}

	return doJobs(jobset);
}




static bool doJob(const CCompositeGeometryManager::TJobSetInternal::const_iterator& iter)
{
	CCompositeGeometryManager::CPart* output = iter->first.get();
	const CCompositeGeometryManager::TJobListInternal& L = iter->second;

	assert(!L.empty());

	ASTRA_DEBUG("first mode: %d", L.begin()->eMode);
	bool zero = L.begin()->eMode == CCompositeGeometryManager::MODE_SET;

	size_t outx, outy, outz;
	output->getDims(outx, outy, outz);

	astraCUDA3d::SSubDimensions3D dstdims = getPartSubDims(output);
	ASTRA_DEBUG("dstdims: %d,%d,%d in %d,%d,%d", dstdims.subnx, dstdims.subny, dstdims.subnz, dstdims.nx, dstdims.ny, dstdims.nz);

	if (L.begin()->eType == CCompositeGeometryManager::JOB_NOP) {
		// just zero output?
		if (zero) {
			// TODO: This function shouldn't have to know about this difference
			// between Memory/GPU
			if (output->pData->getStorage()->isMemory()) {
				assert(output->pData->isFloat32Memory());
				for (size_t z = 0; z < outz; ++z) {
					for (size_t y = 0; y < outy; ++y) {
						float* ptr = output->pData->getFloat32Memory();
						ptr += (z + output->subX) * (size_t)output->pData->getHeight() * (size_t)output->pData->getWidth();
						ptr += (y + output->subY) * (size_t)output->pData->getWidth();
						ptr += output->subX;
						memset(ptr, 0, sizeof(float) * outx);
					}
				}
			} else {
				assert(output->pData->getStorage()->isGPU());
				CDataGPU *gpuMem = dynamic_cast<CDataGPU *>(output->pData->getStorage());
				assert(gpuMem);
				assert(output->isFull()); // TODO: zero subset?

				zeroGPUMemory(gpuMem->getHandle(), outx, outy, outz);
			}
		}
		return true;
	}

	CGPUMemory *dstMem = createGPUMemoryHandler(output->pData);
	// Use a unique_ptr to delete dstMem when we return (early)
	std::unique_ptr<CGPUMemory> dstUniquePtr{dstMem};

	bool ok = dstMem->allocateGPUMemory(outx, outy, outz, zero ? astraCUDA3d::INIT_ZERO : astraCUDA3d::INIT_NO);
	// TODO: cleanup and return error code after any error
	//
	// TODO: Make the memory handler a stack object and clean up afterwards?
	//
	if (!ok) {
		ASTRA_ERROR("Error allocating GPU memory");
		return false;
	}

	if (!zero) {
		// instead of zeroing output memory, copy from host
		ok = dstMem->copyToGPUMemory(dstdims);
		if (!ok) {
			ASTRA_ERROR("Error copying output data to GPU");
			return false;
		}
	}

	for (CCompositeGeometryManager::TJobListInternal::const_iterator i = L.begin(); i != L.end(); ++i) {
		const CCompositeGeometryManager::SJobInternal &j = *i;

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

		CGPUMemory *srcMem = createGPUMemoryHandler(j.pInput->pData);
		// Use a unique_ptr to delete srcMem when we return (early)
		std::unique_ptr<CGPUMemory> srcUniquePtr{srcMem};

		astraCUDA3d::SSubDimensions3D srcdims = getPartSubDims(j.pInput.get());

		ok = srcMem->allocateGPUMemory(inx, iny, inz, astraCUDA3d::INIT_NO);
		if (!ok) {
			ASTRA_ERROR("Error allocating GPU memory");
			return false;
		}

		ok = srcMem->copyToGPUMemory(srcdims);
		if (!ok) {
			ASTRA_ERROR("Error copying input data to GPU");
			return false;
		}

		switch (j.eType) {
		case CCompositeGeometryManager::JOB_FP:
		{
			assert(dynamic_cast<CCompositeGeometryManager::CVolumePart*>(j.pInput.get()));
			assert(dynamic_cast<CCompositeGeometryManager::CProjectionPart*>(output));

			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing FP");

			ok = astraCUDA3d::FP(((CCompositeGeometryManager::CProjectionPart*)output)->pGeom, dstMem->hnd, ((CCompositeGeometryManager::CVolumePart*)j.pInput.get())->pGeom, srcMem->hnd, detectorSuperSampling, projKernel);
			if (!ok) {
				ASTRA_ERROR("Error performing sub-FP");
				return false;
			}
			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: FP done");
		}
		break;
		case CCompositeGeometryManager::JOB_BP:
		{
			assert(dynamic_cast<CCompositeGeometryManager::CVolumePart*>(output));
			assert(dynamic_cast<CCompositeGeometryManager::CProjectionPart*>(j.pInput.get()));

			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing BP");

			ok = astraCUDA3d::BP(((CCompositeGeometryManager::CProjectionPart*)j.pInput.get())->pGeom, srcMem->hnd, ((CCompositeGeometryManager::CVolumePart*)output)->pGeom, dstMem->hnd, voxelSuperSampling, projKernel);
			if (!ok) {
				ASTRA_ERROR("Error performing sub-BP");
				return false;
			}
			ASTRA_DEBUG("CCompositeGeometryManager::doJobs: BP done");
		}
		break;
		case CCompositeGeometryManager::JOB_FDK:
		{
			assert(dynamic_cast<CCompositeGeometryManager::CVolumePart*>(output));
			assert(dynamic_cast<CCompositeGeometryManager::CProjectionPart*>(j.pInput.get()));

			float fOutputScale = srcdims.subny;
			fOutputScale /= srcdims.ny;

			if (j.FDKSettings.bShortScan && srcdims.subny != srcdims.ny) {
				ASTRA_ERROR("CCompositeGeometryManager::doJobs: shortscan FDK unsupported for this data size currently");
				return false;
			} else if (srcdims.subx) {
				ASTRA_ERROR("CCompositeGeometryManager::doJobs: data too large for FDK");
				return false;
			} else {
				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: doing FDK");

				ok = astraCUDA3d::FDK(((CCompositeGeometryManager::CProjectionPart*)j.pInput.get())->pGeom, srcMem->hnd, ((CCompositeGeometryManager::CVolumePart*)output)->pGeom, dstMem->hnd, j.FDKSettings.bShortScan, j.FDKSettings.filterConfig, fOutputScale );
				if (!ok) {
					ASTRA_ERROR("Error performing sub-FDK");
					return false;
				}
				ASTRA_DEBUG("CCompositeGeometryManager::doJobs: FDK done");
			}
		}
		break;
		default:
			ASTRA_ERROR("Internal error: invalid CGM job type");
			return false;
		}

		// srcUniquePtr goes out of scope here, freeing srcMem
	}

	ok = dstMem->copyFromGPUMemory(dstdims);
	if (!ok) {
	       ASTRA_ERROR("Error copying output data from GPU");
	       return false;
	}

	// dstUniquePtr goes out of scope here, freeing dstMem

	return true;
}


class WorkQueue {
public:
	WorkQueue(CCompositeGeometryManager::TJobSetInternal &_jobs) : m_jobs(_jobs), failed(false) {
		m_iter = m_jobs.begin();
	}
	void flag_failure() {
		failed = true;
	}
	bool has_failed() const {
		return failed;
	}
	bool receive(CCompositeGeometryManager::TJobSetInternal::const_iterator &i) {
		if (failed)
			return false;

		lock();

		if (m_iter == m_jobs.end()) {
			unlock();
			return false;
		}

		i = m_iter++;

		unlock();

		return true;
	}
	void lock() {
		m_mutex.lock();
	}
	void unlock() {
		m_mutex.unlock();
	}

private:
	CCompositeGeometryManager::TJobSetInternal &m_jobs;
	CCompositeGeometryManager::TJobSetInternal::const_iterator m_iter;
	std::atomic<bool> failed;
	std::mutex m_mutex;
};

struct WorkThreadInfo {
	WorkQueue* m_queue;
	unsigned int m_iGPU;
};

void runEntries(WorkThreadInfo* info)
{
	ASTRA_DEBUG("Launching thread on GPU %d", info->m_iGPU);
	CCompositeGeometryManager::TJobSetInternal::const_iterator i;
	while (info->m_queue->receive(i)) {
		ASTRA_DEBUG("Running block on GPU %d", info->m_iGPU);
		astraCUDA3d::setGPUIndex(info->m_iGPU);
		if (!doJob(i)) {
			ASTRA_DEBUG("Thread on GPU %d reporting failure", info->m_iGPU);
			info->m_queue->flag_failure();
		}
	}
	ASTRA_DEBUG("Finishing thread on GPU %d", info->m_iGPU);
}

void runWorkQueue(WorkQueue &queue, const std::vector<int> & iGPUIndices) {
	int iThreadCount = iGPUIndices.size();

	std::vector<WorkThreadInfo> infos;
	std::vector<std::thread*> threads;
	infos.resize(iThreadCount);
	threads.resize(iThreadCount);
	ASTRA_DEBUG("Thread count %d", iThreadCount);

	for (int i = 0; i < iThreadCount; ++i) {
		infos[i].m_queue = &queue;
		infos[i].m_iGPU = iGPUIndices[i];
		threads[i] = new std::thread(runEntries, &infos[i]);
	}

	// Wait for them to finish
	for (int i = 0; i < iThreadCount; ++i) {
		threads[i]->join();
		delete threads[i];
		threads[i] = 0;
	}
}


void CCompositeGeometryManager::setGPUIndices(const std::vector<int>& GPUIndices)
{
	m_GPUIndices = GPUIndices;
}

bool CCompositeGeometryManager::doJobs(TJobList &jobs)
{
	// Convert job list into (internal) job set.
	// We are assuming the outputs are disjoint, so each output is
	// associated with a single job.

	TJobSetInternal jobset;

	for (SJob &job : jobs) {
		TJobListInternal newjobs;
		SJobInternal newjob{std::move(job.pInput)};
		newjob.eMode = job.eMode;
		newjob.pProjector = job.pProjector;
		newjob.FDKSettings = job.FDKSettings;
		newjob.eType = job.eType;
		newjobs.push_back(std::move(newjob));

		jobset.push_back(std::make_pair(std::move(job.pOutput), std::move(newjobs)));
	}
	return doJobs(jobset);
}

bool CCompositeGeometryManager::doJobs(TJobSetInternal &jobset)
{
	// TODO: Proper clean up if substeps fail (Or as proper as possible)

	ASTRA_DEBUG("CCompositeGeometryManager::doJobs starting");

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
	TJobSetInternal split;
	splitJobs(jobset, maxSize, div, split);
	jobset.clear();

	if (m_GPUIndices.size() <= 1) {

		// Run jobs
		ASTRA_DEBUG("Running single-threaded");

		if (!m_GPUIndices.empty())
			astraCUDA3d::setGPUIndex(m_GPUIndices[0]);

		for (TJobSetInternal::const_iterator iter = split.begin(); iter != split.end(); ++iter) {
			if (!doJob(iter)) {
				ASTRA_DEBUG("doJob failed, aborting");
				return false;
			}
		}

	} else {

		ASTRA_DEBUG("Running multi-threaded");

		WorkQueue wq(split);

		runWorkQueue(wq, m_GPUIndices);

		if (wq.has_failed())
			return false;

	}

	ASTRA_DEBUG("CCompositeGeometryManager::doJobs done");

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
	ASTRA_DEBUG("%s", ss.c_str());
	ASTRA_DEBUG("Memory: %zu", params.memory);
}


}

#endif
