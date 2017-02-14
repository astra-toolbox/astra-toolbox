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
	//~ D_x = 0;
	D_xTilde = 0;
	D_xold = 0;
	D_dualp = 0;
	D_dualq = 0;
	D_gradTmp = 0; //
	D_gradTmp2 = 0; //
	D_sigma = 0;
	D_tau = 0;

	D_minMaskData = 0;
	D_maxMaskData = 0;

	normFactor = 1.2;
	fRegularization = 1.0f;
	freeMinMaxMasks = false;

	dimsGrad = dims;
}


TV::~TV()
{
	reset();
}

void TV::reset()
{
	cudaFree(D_projData);
	//~ cudaFree(D_x);
	cudaFree(D_xTilde);
	cudaFree(D_xold);
	cudaFree(D_dualp);
	cudaFree(D_dualq);
	cudaFree(D_gradTmp); //
	cudaFree(D_gradTmp2); //
	cudaFree(D_tau);
	cudaFree(D_sigma);

	if (freeMinMaxMasks) {
		cudaFree(D_minMaskData);
		cudaFree(D_maxMaskData);
	}

	D_projData = 0;
	//~ D_x = 0;
	D_xTilde = 0;
	D_xold = 0;
	D_dualp = 0;
	D_dualq = 0;
	D_minMaskData = 0;
	D_maxMaskData = 0;
	D_gradTmp = 0; //
	D_gradTmp2 = 0; //
	D_sigma = 0;
	D_tau = 0;


	freeMinMaxMasks = false;
	useVolumeMask = false;
	useSinogramMask = false;
	fRegularization = 1.0f;

	ReconAlgo::reset();
}

bool TV::init()
{

    allocateProjectionData(D_projData, projPitch, dims);
	zeroProjectionData(D_projData, projPitch, dims);

    allocateVolumeData(D_x, xPitch, dims);
	zeroVolumeData(D_x, xPitch, dims);

    allocateVolumeData(D_xold, xoldPitch, dims);
	zeroVolumeData(D_xold, xoldPitch, dims);

    allocateVolumeData(D_xTilde, xtildePitch, dims);
	zeroVolumeData(D_xTilde, xtildePitch, dims);

	allocateVolumeData(D_tau, tauPitch, dims);
	zeroVolumeData(D_tau, tauPitch, dims);

	allocateProjectionData(D_dualq, dualqPitch, dims);
	zeroProjectionData(D_dualq, dualqPitch, dims);

	allocateProjectionData(D_sigma, sigmaPitch, dims);
	zeroProjectionData(D_sigma, sigmaPitch, dims);

	// if float2 cannot be used, we use a buffer with height*2
    dimsGrad = dims;
    dimsGrad.iVolHeight *= 2;
    allocateVolumeData(D_dualp, dualpPitch, dimsGrad);
	zeroVolumeData(D_dualp, dualpPitch, dimsGrad);
   	// if not cublas
   	allocateVolumeData(D_gradTmp, gradTmpPitch, dimsGrad);
	zeroVolumeData(D_gradTmp, gradTmpPitch, dimsGrad);
	allocateVolumeData(D_gradTmp2, gradTmp2Pitch, dimsGrad);
	zeroVolumeData(D_gradTmp2, gradTmp2Pitch, dimsGrad);

    nIterComputeNorm = 20;
    normFactor = 1.2f;

	// TODO: check if allocations succeeded
	return true;
}


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





__global__ void projLinfKernel(float* dst, float* src, const SDimensions dims, unsigned int pitch, float radius) {
	unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;

	if (gidx < sizeX && gidy < sizeY) {
		unsigned int idx = gidy*pitch+gidx;
		float val_x = src[idx];
		float val_y = src[pitch*sizeY + idx];

		dst[idx] = copysignf(min(fabsf(val_x), radius), val_x);
		dst[pitch*sizeY + idx] = copysignf(min(fabsf(val_y), radius), val_y);
	}
}

bool TV::projLinf(float* D_gradData, float* D_data, unsigned int pitch, float radius) {
	dim3 nBlocks, nThreadsPerBlock;
	nThreadsPerBlock = dim3(threadsPerBlock, threadsPerBlock, 1);
	nBlocks = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

	projLinfKernel<<<nBlocks, nThreadsPerBlock>>>(D_data, D_gradData, dims, pitch, radius);
	return true;
}



__global__ void gradientKernel2D(float* dst, float* src, const SDimensions dims, unsigned int pitch, float alpha, int doUpdate) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;
    float val_x = 0, val_y = 0;

    if (gidx < sizeX && gidy < sizeY) {
        if (gidx == sizeX-1) val_y = 0;
        else val_y = src[(gidy)*pitch+gidx+1] - src[gidy*pitch+gidx];
        if (gidy == sizeY-1) val_x = 0;
        else val_x = src[(gidy+1)*pitch+gidx] - src[gidy*pitch+gidx];

        if (doUpdate) {
            val_x = alpha*val_x + dst[gidy*pitch+gidx];
            val_y = alpha*val_y + dst[pitch*sizeY + gidy*pitch+gidx];
        }

        dst[(gidy)*pitch+gidx] = val_x;
        dst[pitch*sizeY + (gidy)*pitch+gidx] = val_y;
    }
}


// gradientOperator(dst, src, alpha, 0)  computes  dst = gradient(src)
// gradientOperator(dst, src, alpha, 1)  computes  dst = dst + alpha*gradient(src)
bool TV::gradientOperator(float* D_gradData, float* D_data, unsigned int pitch, float alpha, int doUpdate) {
    dim3 nBlocks, nThreadsPerBlock;
    nThreadsPerBlock = dim3(threadsPerBlock, threadsPerBlock, 1);
    nBlocks = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

    gradientKernel2D<<<nBlocks, nThreadsPerBlock>>>(D_gradData, D_data, dims, pitch, alpha, doUpdate);
    return true;
}


__global__ void divergenceKernel2D(float* dst, float* src, const SDimensions dims, unsigned int pitch, float alpha, int doUpdate) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;
    float val_x = 0, val_y = 0;

    if (gidx < sizeX && gidy < sizeY) {
        if (gidx == 0) val_y = src[(gidy)*pitch+gidx];
        else val_y = src[pitch*sizeY + (gidy)*pitch+gidx] - src[pitch*sizeY +  (gidy)*pitch+gidx-1];
        if (gidy == 0) val_x = src[(gidy)*pitch+gidx];
        else val_x = src[(gidy)*pitch+gidx] - src[(gidy-1)*pitch+gidx];

        if (doUpdate) dst[(gidy)*pitch+gidx] += alpha*(val_x + val_y);
        else dst[(gidy)*pitch+gidx] = val_x + val_y;
    }
}
// divergenceOperator(dst, src, alpha, 0)  computes  dst = div(src)
// divergenceOperator(dst, src, alpha, 1)  computes  dst = dst + alpha*div(src)
bool TV::divergenceOperator(float* D_data, float* D_gradData, unsigned int pitch, float alpha, int doUpdate) {
    dim3 nBlocks, nThreadsPerBlock;
    nThreadsPerBlock = dim3(threadsPerBlock, threadsPerBlock, 1);
    nBlocks = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

    divergenceKernel2D<<<nBlocks, nThreadsPerBlock>>>(D_data, D_gradData, dims, pitch, alpha, doUpdate);
    return true;
}


__global__ void signKernel2D(float* dst, float* src, const SDimensions dims, unsigned int pitch, int nz) {
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int sizeX = dims.iVolWidth, sizeY = dims.iVolHeight;
    unsigned int idx = gidy*pitch + gidx;
    if (gidx < sizeX && gidy < sizeY) {
		dst[idx] = copysignf(1, src[idx]);
		if (nz > 1) for (int i = 1; i < nz; i++) {
			dst[i*pitch*sizeY + idx] = copysignf(1, src[i*pitch*sizeY + idx]);
		}
	}
}

// signOperator(dst, src, 1) computes dst = sign(src).
// If the last parameter is greater than 1, it means that there are several buffers
bool TV::signOperator(float* D_dst, float* D_src, unsigned int pitch, int nz) {
    dim3 nBlocks, nThreadsPerBlock;
    nThreadsPerBlock = dim3(threadsPerBlock, threadsPerBlock, 1);
    nBlocks = dim3(iDivUp(dims.iVolWidth, threadsPerBlock), iDivUp(dims.iVolHeight, threadsPerBlock), 1);

    signKernel2D<<<nBlocks, nThreadsPerBlock>>>(D_dst, D_src, dims, pitch, nz);
    return true;
}




__global__ void updateDualq1Kernel(float* out, unsigned int outPitch,
								  float* in1, unsigned int in1Pitch,
								  float* in2, unsigned int in2Pitch,
								  const SDimensions dims)
{
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int sizeX = dims.iProjDets, sizeY = dims.iProjAngles;
    if (gidx < sizeX && gidy < sizeY) {
		out[gidy*outPitch + gidx] = out[gidy*outPitch + gidx]/in1[gidy*in1Pitch + gidx] - in2[gidy*in2Pitch + gidx];
	}
}

/// Compute out = out/in1 - in2  (sinogram-like buffers)
bool TV::callUpdateDualq1(float* D_out, unsigned int outPitch,
					     float* D_in1, unsigned int in1Pitch,
					     float* D_in2, unsigned int in2Pitch)
{
    dim3 nBlocks, nThreadsPerBlock;
    nThreadsPerBlock = dim3(threadsPerBlock, threadsPerBlock, 1);
    nBlocks = dim3(iDivUp(dims.iProjDets, threadsPerBlock), iDivUp(dims.iProjAngles, threadsPerBlock), 1);

    updateDualq1Kernel<<<nBlocks, nThreadsPerBlock>>>(D_out, outPitch, D_in1, in1Pitch, D_in2, in2Pitch, dims);
    return true;
}







__global__ void updateDualq2Kernel(float* out, unsigned int outPitch,
								  float* in, unsigned int inPitch,
								  const SDimensions dims)
{
    unsigned int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int sizeX = dims.iProjDets, sizeY = dims.iProjAngles;
    if (gidx < sizeX && gidy < sizeY) {
		// TODO: check ?
		out[gidy*outPitch + gidx] = in[gidy*inPitch + gidx]*out[gidy*outPitch + gidx] / (1.0f + in[gidy*inPitch + gidx]);
	}
}

/// Compute out = in*out/(1+in)  (sinogram-like buffers)
bool TV::callUpdateDualq2(float* D_out, unsigned int outPitch,
					     float* D_in, unsigned int inPitch)
{
    dim3 nBlocks, nThreadsPerBlock;
    nThreadsPerBlock = dim3(threadsPerBlock, threadsPerBlock, 1);
    nBlocks = dim3(iDivUp(dims.iProjDets, threadsPerBlock), iDivUp(dims.iProjAngles, threadsPerBlock), 1);

    updateDualq2Kernel<<<nBlocks, nThreadsPerBlock>>>(D_out, outPitch, D_in, inPitch, dims);
    return true;
}







// Compute the norm of the operator K = [grad, P]
// using ||K|| = sqrt(max_eigen(K^T * K))
float TV::computeOperatorNorm() {
    float norm = -1.0f;

    zeroVolumeData(D_x, xPitch, dims);
    callBP(D_x, xPitch, D_sinoData, sinoPitch, 1.0f);

    // power method for computing max eigenval of P^T P
    for (unsigned int iter = 0 ; iter < nIterComputeNorm; ++iter) {
        // x := P^T(P(x)) - div(grad(x))
        zeroProjectionData(D_dualq, dualqPitch, dims);
        callFP(D_x, xPitch, D_dualq, dualqPitch, 1.0f);
        zeroVolumeData(D_x, xPitch, dims);
        callBP(D_x, xPitch, D_dualq, dualqPitch, 1.0f);
        gradientOperator(D_dualp, D_x, xPitch, 1.0, 0);
        divergenceOperator(D_x, D_dualp, xPitch, -1.0f, 1); // CHECKME

        // Compute norm and scale x
        norm = dotProduct2D(D_x, xPitch, dims.iVolWidth, dims.iVolHeight); // TODO: check
        norm = sqrt(norm);
        processVol<opMul>(D_x, 1.0f/norm, xPitch, dims);
    }
    //
    zeroVolumeData(D_dualp, dualpPitch, dimsGrad);
    zeroVolumeData(D_x, xPitch, dims);
	zeroProjectionData(D_dualq, dualqPitch, dims);
    //
    if (norm < 0) return -1.0f;     // something went wrong
    else return sqrt(norm);
}










/**
 * Compute the diagonal preconditioners "Sigma" and "Tau" described in [1].
 *
 * Here alpha = 1 is used in [1], as the sum of the projection operator
 * absolute values consists in projecting/backprojecting "ones".
 *
 * Sigma (resp. Tau) is the sum of the operator absolute values
 * along the columns (resp. lines).
 * The operator considered here is K = [D, P]^T where D is the discrete
 * gradient operator, and P is the projection operator:
 *
 *  	| D |		  	      | Sigma_D |
 * K =  | P |   -->   Sigma = | Sigma_P |
 *
 *        |
 * 	      v
 *
 *	 Tau = Tau_D + Tau_P
 *
 * For computational convenience (see iterate()), the returned Sigma
 * is Sigma_P.
 *
 * [1] Pock, Thomas and Chambolle, Antonin, "Diagonal preconditioning for
 * first order primal-dual algorithms in convex optimization", 2011,
 * International Conference on Computer Vision, pp 1762-1769
 */
bool TV::computeDiagonalPreconditioners() {


	// Project a slice of "ones" to get the sum of projector columns
	zeroVolumeData(D_xTilde, xtildePitch, dims);
	processVol<opAdd>(D_xTilde, 1.0, xtildePitch, dims);
	zeroProjectionData(D_sigma, sigmaPitch, dims);
	callFP(D_xTilde, xtildePitch, D_sigma, sigmaPitch, 1);
	//~ processSino<opAdd>(D_sigma, 2.0, sigmaPitch, dims); // sum of columns of gradient is 2
	processSino<opInvert>(D_sigma, sigmaPitch, dims);

	// Backproject a sinogram of "ones" to get the sum of projector lines
	zeroProjectionData(D_dualq, dualqPitch, dims);
	processSino<opAdd>(D_dualq, 1.0, dualqPitch, dims);
	callBP(D_tau, tauPitch, D_dualq, dualqPitch, 1);
	processVol<opAdd>(D_tau, 2.0, tauPitch, dims); // sum of lines of gradient is 2
	processVol<opInvert>(D_tau, tauPitch, dims);

	//
	zeroVolumeData(D_xTilde, xtildePitch, dims);
	zeroProjectionData(D_dualq, dualqPitch, dims);
	//

	return true;
}




bool TV::chambollepock_preconditioned(unsigned int iterations)
{
    // Compute the primal and dual steps
    float theta = 1.;				  	// C-P relaxation parameter
    float sigma_grad = 0.5;				// Diagonal preconditioner, gradient part
	computeDiagonalPreconditioners();

	// iteration
	for (unsigned int iter = 0; iter < iterations; ++iter) {

		// Update dual variables
		// ----------------------
		// p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
		gradientOperator(D_dualp, D_xTilde, dualpPitch, sigma_grad, 1);
		projLinf(D_dualp, D_dualp, dualpPitch, fRegularization); // *sigma
		// q = (q + sigma*P(x_tilde) - sigma*data)/(1.0 + sigma)
		// q = (q/sigma + P(x_tilde) - data)*sigma / (1+sigma)
		callUpdateDualq1(D_dualq, dualqPitch, D_sigma, sigmaPitch, 		// q = q/sigma - data
						 D_sinoData, sinoPitch);
		callFP(D_xTilde, xtildePitch, D_dualq, dualqPitch, 1);      	// q += P(xtilde)
		callUpdateDualq2(D_dualq, dualqPitch, D_sigma, sigmaPitch);		// q *= sigma/(1+sigma)

		// Update primal variables
		// ------------------------
		duplicateVolumeData(D_xold, D_x, xPitch, dims);
		// x = x + tau*div(p) - tau*P^T(q)
		divergenceOperator(D_xTilde, D_dualp, xtildePitch, 0, 0);		// tmp = div(p)
		callBP(D_xTilde, xtildePitch, D_dualq, dualqPitch, -1);			// tmp -= P^T(q)
		processVol<opMul>(D_xTilde, D_tau, xtildePitch, dims);			// tmp *= tau
		processVol<opAddScaled>(D_x, D_xTilde, 1.0, xPitch, dims);		// x += tmp

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
        processVol<opAddScaled>(D_xTilde, D_xold, -theta, xtildePitch, dims);
        // TODO: this in two steps ?

	}

	 duplicateVolumeData(D_volumeData, D_x, volumePitch, dims);

	return true;
}



bool TV::chambollepock(unsigned int iterations)
{
    // Compute the primal and dual steps, for non-preconditionned CP
    float L = computeOperatorNorm();  //TODO: abort if norm is negative
    float sigma = 1.0f/L;       	  // dual step
    float tau = 1.0f/L;         	  // primal step
    float theta = 1.;				  // C-P relaxation parameter

	// iteration
	for (unsigned int iter = 0; iter < iterations; ++iter) {

		// Update dual variables
		// ----------------------
		// p = proj_linf(p + sigma*gradient(x_tilde), Lambda)
		gradientOperator(D_dualp, D_xTilde, dualpPitch, sigma, 1);
		projLinf(D_dualp, D_dualp, dualpPitch, fRegularization); // *sigma
		// q = (q + sigma*P(x_tilde) - sigma*data)/(1.0 + sigma)
        callFP(D_xTilde, xtildePitch, D_dualq, dualqPitch, sigma);          // q = q + sigma*P(xtilde)
        processSino<opAddScaled>(D_dualq, D_sinoData, -sigma, 				// q -= sigma*data
								 dualqPitch, dims);
        processSino<opMul>(D_dualq, 1.0f/(1.0f+sigma), dualqPitch, dims);   // q /= 1+sigma

		// Update primal variables
		// ------------------------
		duplicateVolumeData(D_xold, D_x, xPitch, dims);
		// x = x + tau*div(p) - tau*P^T(q)
		divergenceOperator(D_x, D_dualp, xPitch, tau, 1);					// x = x + tau*div(p)
		callBP(D_x, xPitch, D_dualq, dualqPitch, -tau);		  				// x += (-tau)*P^T(q)

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
        processVol<opAddScaled>(D_xTilde, D_xold, -theta, xtildePitch, dims);
        // TODO: this in two steps ?

	}

	 duplicateVolumeData(D_volumeData, D_x, volumePitch, dims);

	return true;
}



bool TV::iterate(unsigned int iterations)
{
	/**
	 If iProjDets > iVolWidth, the projection of a slice of "ones" contains zero near the boundaries.
	 Thus, inverting this projection (needed for the preconditioned version) is not a good idea.
	 A work-around would be to invert only the sinogram values at locations
	 falling in the slice support when backprojected.
	 However, the center of rotation must be known. We can assume that it is in the center of the slice,
	 but it still makes an assumption on the geometry.
	 Thus, if iProjDets > iVolWidth, we fall back to the not-preconditioned algorithm.
	**/
	if (dims.iProjDets > dims.iVolWidth) {
		return chambollepock(iterations);
	}
	else {
		return chambollepock_preconditioned(iterations);
	}
}



/// Compute  0.5 * ||P(x) - data||_2^2  + Lambda*TV(x)
float TV::computeDiffNorm()
{
	// copy sinogram to projection data
	duplicateProjectionData(D_projData, D_sinoData, projPitch, dims);

	// do FP, subtracting projection from sinogram
	callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);

	// compute norm of D_projData
	float l2 = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);
	l2 *= 0.5;

	// cublasSasum() would be ideal. If it cannot be used,
	// the only solution is to use a dot product between grad(x) and sign(grad(x)),
	// but it entails two extra gradient buffers
	gradientOperator(D_gradTmp, D_volumeData, gradTmpPitch, 0, 0);
	signOperator(D_gradTmp2, D_gradTmp, gradTmpPitch, 2);
	float l1 = dotProduct2D(D_gradTmp, gradTmpPitch, dims.iVolWidth, dims.iVolHeight);

	return l2 + fRegularization*l1;
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

