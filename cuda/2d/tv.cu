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

#include <cstdio>
#include <cassert>

#include "tv.h"
#include "util.h"
#include "arith.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

namespace astraCUDA {


// optimization parameters
static const unsigned int threadsPerBlock = 16;


static int iDivUp(int a, int b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}




TV::TV() : ReconAlgo()
{
	D_projData = 0;
	D_tmpData = 0;

	D_lineWeight = 0;
	D_pixelWeight = 0;

	D_minMaskData = 0;
	D_maxMaskData = 0;

	fRelaxation = 1.0f;

	freeMinMaxMasks = false;
}


TV::~TV()
{
	reset();
}

void TV::reset()
{
	cudaFree(D_projData);
	cudaFree(D_tmpData);
	cudaFree(D_lineWeight);
	cudaFree(D_pixelWeight);
	if (freeMinMaxMasks) {
		cudaFree(D_minMaskData);
		cudaFree(D_maxMaskData);
	}

	D_projData = 0;
	D_tmpData = 0;

	D_lineWeight = 0;
	D_pixelWeight = 0;

	freeMinMaxMasks = false;
	D_minMaskData = 0;
	D_maxMaskData = 0;

	useVolumeMask = false;
	useSinogramMask = false;

	fRelaxation = 1.0f;

	ReconAlgo::reset();
}

bool TV::init()
{

    allocateVolumeData(D_x, xPitch, dims);
	zeroVolumeData(D_x, xPitch, dims);

    allocateVolumeData(D_xold, xoldPitch, dims);
	zeroVolumeData(D_xold, xoldPitch, dims);

    allocateVolumeData(sliceTmp, tmpPitch, dims);
	zeroVolumeData(sliceTmp, tmpPitch, dims);

    allocateVolumeData(D_xTilde, xtildePitch, dims);
	zeroVolumeData(D_pixelWeight, xtildePitch, dims);

    int nels = dims.iVolWidth * dims.iVolHeight;
    cudaMalloc(&D_dualp, nels*sizeof(float2));
    cudaMemset(D_dualp, 0, nels*sizeof(float2));

	allocateProjectionData(D_sinoTmp, sinoTmpPitch, dims);
	zeroProjectionData(D_sinoTmp, sinoTmpPitch, dims);

	allocateProjectionData(D_dualq, dualqPitch, dims);
	zeroProjectionData(D_dualq, dualqPitch, dims);


    nIterComputeNorm = 20;
    normFactor = 1.2f;


	// TODO: check if allocations succeeded
	return true;
}

/*
bool TV::setMinMaxMasks(float* D_minMaskData_, float* D_maxMaskData_,
	                      unsigned int iPitch)
{
	D_minMaskData = D_minMaskData_;
	D_maxMaskData = D_maxMaskData_;
	minMaskPitch = iPitch;
	maxMaskPitch = iPitch;

	freeMinMaxMasks = false;
	return true;
}

bool TV::uploadMinMaxMasks(const float* pfMinMaskData, const float* pfMaxMaskData,
	                         unsigned int iPitch)
{
	freeMinMaxMasks = true;
	bool ok = true;
	if (pfMinMaskData) {
		allocateVolumeData(D_minMaskData, minMaskPitch, dims);
		ok = copyVolumeToDevice(pfMinMaskData, iPitch,
		                        dims,
		                        D_minMaskData, minMaskPitch);
	}
	if (!ok)
		return false;

	if (pfMaxMaskData) {
		allocateVolumeData(D_maxMaskData, maxMaskPitch, dims);
		ok = copyVolumeToDevice(pfMaxMaskData, iPitch,
		                        dims,
		                        D_maxMaskData, maxMaskPitch);
	}
	if (!ok)
		return false;

	return true;
}
*/






__global__ void projLinfKernel(float* dst, float2* src, const SDimensions dims, float radius) {
	int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;

	if (gidx < sizeX && gidy < sizeY) {
		int idx = gidy*sizeX+gidx;
		float val_x = src[idx].x;
		float val_y = src[idx].y;

		dst[idx].x = copysignf(min(fabsf(val_x), radius), val_x);
		dst[idx].y = copysignf(min(fabsf(val_y), radius), val_y);
	}
}

bool TV::projLinf(float2* D_gradData, float* D_data, float radius) {
	dim3 nBlocks, nThreadsPerBlock;
	blk = dim3(threadsPerBlock, threadsPerBlock, 1);
	grd = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

	projLinfKernel<<<grd, blk>>>(D_Data, D_gradData, dims, radius)
	return true;
}



__global__ void gradientKernel2D(float2* dst, float* src, const SDimensions dims, float alpha, int doUpdate) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;
    float val_x = 0, val_y = 0;

    if (gidx < sizeX && gidy < sizeY) {
        if (gidx == sizeX-1) val_y = 0;
        else val_y = src[(gidy)*sizeX+gidx+1] - src[gidy*sizeX+gidx];
        if (gidy == sizeY-1) val_x = 0;
        else val_x = src[(gidy+1)*sizeX+gidx] - src[gidy*sizeX+gidx];

        if (doUpdate) {
            val_x = alpha*val_x + dst[gidy*sizeX+gidx].x;
            val_y = alpha*val_y + dst[gidy*sizeX+gidx].y;
        }

        slice_grad[(gidy)*sizeX+gidx].x = val_x;
        slice_grad[(gidy)*sizeX+gidx].y = val_y;
    }
}


// gradientOperator(dst, src, alpha, 0)  computes  dst = gradient(src)
// gradientOperator(dst, src, alpha, 1)  computes  dst = dst + alpha*gradient(src)
bool TV::gradientOperator(float2* D_gradData, float* D_data, float alpha, int doUpdate) {
    dim3 nBlocks, nThreadsPerBlock;
    blk = dim3(threadsPerBlock, threadsPerBlock, 1);
    grd = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

    gradientKernel2D<<<grd, blk>>>(D_gradData, D_data, dims, alpha, doUpdate);
    return True;
}


__global__ void divergenceKernel2D(float* dst, float2* src, const SDimensions dims, float alpha, int doUpdate) {
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;
    float val_x = 0, val_y = 0;

    if (gidx < sizeX && gidy < sizeY) {
        if (gidx == 0) val_y = slice_grad[(gidy)*sizeX+gidx].y;
        else val_y = src[(gidy)*sizeX+gidx].y - src[(gidy)*sizeX+gidx-1].y;
        if (gidy == 0) val_x = slice_grad[(gidy)*sizeX+gidx].x;
        else val_x = src[(gidy)*sizeX+gidx].x - src[(gidy-1)*sizeX+gidx].x;

        if (doUpdate) dst[(gidy)*sizeX+gidx] += alpha*(val_x + val_y);
        else dst[(gidy)*sizeX+gidx] = val_x + val_y;
    }
}


// divergenceOperator(dst, src, alpha, 0)  computes  dst = div(src)
// divergenceOperator(dst, src, alpha, 1)  computes  dst = dst + alpha*div(src)
bool TV::divergenceOperator(float* D_data, float2* D_gradData, float alpha, int doUpdate) {
    dim3 nBlocks, nThreadsPerBlock;
    blk = dim3(threadsPerBlock, threadsPerBlock, 1);
    grd = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

    divergenceKernel2D<<<grd, blk>>>(D_data, D_gradData, dims, alpha, doUpdate);
    return True;
}



// Compute the norm of the operator K = [grad, P]
// using ||K|| = sqrt(max_eigen(K^T * K))
float TV::computeOperatorNorm() {
    float norm = -1.0f;

    zeroVolumeData(D_sliceTmp, tmpPitch, dims);
    callBP(D_sliceTmp, tmpPitch, D_sinoData, sinoPitch, 1.0f);

    // power method for computing max eigenval of P^T P
    for (unsigned int iter = 0 ; iter < nIterComputeNorm; ++iter) {
        // x := P^T(P(x)) - div(grad(x))
        zeroProjectionData(D_sinoTmp, sinoTmpPitch, dims);
        callFP(D_sliceTmp, tmpPitch, D_sinoTmp, sinoTmpPitch, 1.0f);
        zeroVolumeData(D_sliceTmp, tmpPitch, dims);
        callBP(D_sliceTmp, tmpPitch, D_sinoTmp, sinoTmpPitch, 1.0f);
        gradientOperator(D_dualp, D_sliceTmp);
        divergenceOperator(D_sliceTmp, D_dualp, -1.0f, 1); // TODO: what is computed is div or -div ? In the latter case: put alpha=+1

        // Compute norm and scale x
        norm = dotProduct2D(D_sliceTmp, tmpPitch, dims.iVolWidth, dims.iVolHeight); // TODO: check
        norm = sqrt(norm);
        processVol<opMul>(D_sliceTmp, 1.0f/norm, tmpPitch, dims);
    }
    if (norm < 0) return -1.0f;     // something went wrong
    else return sqrt(norm);
}






// TODO: implement volume mask
// TODO: implement either use_fbp in iterations, or preconditioned CP
bool TV::iterate(unsigned int iterations)
{
    // Compute the primal and dual steps, for non-preconditionned CP
    L = computeOperatorNorm();  //TODO: abort if norm is negative
    float sigma = 1.0f/L;       // dual step
    float tau = 1.0f/L;         // primal step

	// iteration
	for (unsigned int iter = 0; iter < iterations; ++iter) {
		// Update dual variables
		// ----------------------
		// p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
		gradientOperator(D_dualp, D_xTilde, sigma, 1);
		projLinf(D_dualp, lambdaTV)

		// q = (q + sigma*P(x_tilde) - sigma*data)/(1.0 + sigma)
        callFP(D_xTilde, xtildePitch, D_dualq, dualqPitch, sigma);          // q = q + sigma*P(xtilde)
        processSino<opAddScaled>(D_dualq, D_sinoData, -sigma, dims);        // q -= sigma*data
        processSino<opMul>(D_dualq, 1.0f/(1.0f+sigma), dualqPitch, dims);   // q /= 1+sigma

		// Update primal variables
		// ------------------------
		// x = x + tau*div(p) - tau*P^T(q)
        callBP(D_x, xPitch, D_dualq, dualqPitch, -tau);                     // x = x - tau*P^T(q)
        divergenceOperator(D_x, D_dualp, tau, 1);                           // x += tau*div(p)

        // Extra constraints (if any)
        // --------------------------
		if (useMinConstraint)
			processVol<opClampMin>(D_x, fMinConstraint, xPitch, dims);
		if (useMaxConstraint)
			processVol<opClampMax>(D_x, fMaxConstraint, xPitch, dims);
		if (D_minMaskData)
			processVol<opClampMinMask>(D_x, D_minMaskData, xPitch, dims);
		if (D_maxMaskData)
			processVol<opClampMaxMask>(D_x, D_maxMaskData, xPitch, dims);

        // Update step
        // ------------
        // x_tilde = x + theta*(x - x_old) = (1+theta)*x - theta*x_old
        duplicateVolumeData(D_xTilde, D_x, xtildePitch, dims);
        processVol<opMul>(D_xTilde, 1.0f+theta, xtildePitch, dims);
        processVol<opAddScaled>(D_xTilde, -theta, D_xold, xtildePitch, dims);
        // TODO: this in two steps ? maybe x_tilde can be used previously as sliceTmp

	}

	return true;
}

float TV::computeDiffNorm()
{
	// copy sinogram to projection data
	duplicateProjectionData(D_projData, D_sinoData, projPitch, dims);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			duplicateVolumeData(D_tmpData, D_volumeData, volumePitch, dims);
			processVol<opMul>(D_tmpData, D_maskData, tmpPitch, dims);
			callFP(D_tmpData, tmpPitch, D_projData, projPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);
	}


	// compute norm of D_projData

	float s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	return sqrt(s);
}


bool doTV(float* D_volumeData, unsigned int volumePitch,
            float* D_sinoData, unsigned int sinoPitch,
            float* D_maskData, unsigned int maskPitch,
            const SDimensions& dims, const float* angles,
            const float* TOffsets, unsigned int iterations)
{
	TV tv;
	bool ok = true;

	ok &= tv.setGeometry(dims, angles);
	if (D_maskData)
		ok &= tv.enableVolumeMask();
	if (TOffsets)
		ok &= tv.setTOffsets(TOffsets);

	if (!ok)
		return false;

	ok = tv.init();
	if (!ok)
		return false;

	if (D_maskData)
		ok &= tv.setVolumeMask(D_maskData, maskPitch);

	ok &= tv.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);
	if (!ok)
		return false;

	ok = tv.iterate(iterations);

	return ok;
}

}

#ifdef STANDALONE

using namespace astraCUDA;

int main()
{
	float* D_volumeData;
	float* D_sinoData;

	SDimensions dims;
	dims.iVolWidth = 1024;
	dims.iVolHeight = 1024;
	dims.iProjAngles = 512;
	dims.iProjDets = 1536;
	dims.fDetScale = 1.0f;
	dims.iRaysPerDet = 1;
	unsigned int volumePitch, sinoPitch;

	allocateVolume(D_volumeData, dims.iVolWidth, dims.iVolHeight, volumePitch);
	zeroVolume(D_volumeData, volumePitch, dims.iVolWidth, dims.iVolHeight);
	printf("pitch: %u\n", volumePitch);

	allocateVolume(D_sinoData, dims.iProjDets, dims.iProjAngles, sinoPitch);
	zeroVolume(D_sinoData, sinoPitch, dims.iProjDets, dims.iProjAngles);
	printf("pitch: %u\n", sinoPitch);

	unsigned int y, x;
	float* sino = loadImage("sino.png", y, x);

	float* img = new float[dims.iVolWidth*dims.iVolHeight];

	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_sinoData, sinoPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	TV tv;

	tv.setGeometry(dims, angle);
	tv.init();

	tv.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);

	tv.iterate(25);


	delete[] angle;

	copyVolumeFromDevice(img, dims.iVolWidth, dims, D_volumeData, volumePitch);

	saveImage("vol.png",dims.iVolHeight,dims.iVolWidth,img);

	return 0;
}
#endif

