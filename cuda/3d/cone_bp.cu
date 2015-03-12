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
#include <iostream>
#include <list>

#include <cuda.h>
#include "util3d.h"

#ifdef STANDALONE
#include "cone_fp.h"
#include "testutil.h"
#endif

#include "dims3d.h"

typedef texture<float, 3, cudaReadModeElementType> texture3D;

static texture3D gT_coneProjTexture;

namespace astraCUDA3d {

#define ZSIZE 6
static const unsigned int g_volBlockZ = ZSIZE;

static const unsigned int g_anglesPerBlock = 32;
static const unsigned int g_volBlockX = 16;
static const unsigned int g_volBlockY = 32;

static const unsigned g_MaxAngles = 1024;

__constant__ float gC_C[12*g_MaxAngles];

bool bindProjDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_coneProjTexture.addressMode[0] = cudaAddressModeBorder;
	gT_coneProjTexture.addressMode[1] = cudaAddressModeBorder;
	gT_coneProjTexture.addressMode[2] = cudaAddressModeBorder;
	gT_coneProjTexture.filterMode = cudaFilterModeLinear;
	gT_coneProjTexture.normalized = false;

	cudaBindTextureToArray(gT_coneProjTexture, array, channelDesc);

	// TODO: error value?

	return true;
}


//__launch_bounds__(32*16, 4)
__global__ void dev_cone_BP(void* D_volData, unsigned int volPitch, int startAngle,
                            int angleOffset, const astraCUDA3d::SDimensions3D dims,
                            float fOutputScale)
{
	float* volData = (float*)D_volData;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles - angleOffset)
		endAngle = dims.iProjAngles - angleOffset;

	// threadIdx: x = rel x
	//            y = rel y

	// blockIdx:  x = x + y
	//            y = z



	const int X = blockIdx.x % ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockX + threadIdx.x;
	const int Y = blockIdx.x / ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockY + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.y * g_volBlockZ;
	const float fX = X - 0.5f*dims.iVolX + 0.5f;
	const float fY = Y - 0.5f*dims.iVolY + 0.5f;
	const float fZ = startZ - 0.5f*dims.iVolZ + 0.5f;

	float Z[ZSIZE];
	for(int i=0; i < ZSIZE; i++)
		Z[i] = 0.0f;


	{
		float fAngle = startAngle + angleOffset + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
		{
			float4 fCu  = make_float4(gC_C[12*angle+0], gC_C[12*angle+1], gC_C[12*angle+2], gC_C[12*angle+3]);
			float4 fCv  = make_float4(gC_C[12*angle+4], gC_C[12*angle+5], gC_C[12*angle+6], gC_C[12*angle+7]);
			float4 fCd  = make_float4(gC_C[12*angle+8], gC_C[12*angle+9], gC_C[12*angle+10], gC_C[12*angle+11]);

			float fUNum = fCu.w + fX * fCu.x + fY * fCu.y + fZ * fCu.z;
			float fVNum = fCv.w + fX * fCv.x + fY * fCv.y + fZ * fCv.z;
			float fDen  = fCd.w + fX * fCd.x + fY * fCd.y + fZ * fCd.z;

			float fU,fV, fr;

			for (int idx = 0; idx < ZSIZE; idx++)
			{
				fr = __fdividef(1.0f, fDen);
				fU = fUNum * fr;
				fV = fVNum * fr;
				float fVal = tex3D(gT_coneProjTexture, fU, fAngle, fV);
				Z[idx] += fVal; // fr*fr*fVal;

				fUNum += fCu.z;
				fVNum += fCv.z;
				fDen  += fCd.z;
			}
		}
	}

	int endZ = ZSIZE;
	if (endZ > dims.iVolZ - startZ)
		endZ = dims.iVolZ - startZ;

	for(int i=0; i < endZ; i++)
		volData[((startZ+i)*dims.iVolY+Y)*volPitch+X] += Z[i] * fOutputScale;
} //End kernel



// supersampling version
__global__ void dev_cone_BP_SS(void* D_volData, unsigned int volPitch, int startAngle, int angleOffset, const SDimensions3D dims, float fOutputScale)
{
	float* volData = (float*)D_volData;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles - angleOffset)
		endAngle = dims.iProjAngles - angleOffset;

	// threadIdx: x = rel x
	//            y = rel y

	// blockIdx:  x = x + y
    //            y = z


	// TO TRY: precompute part of detector intersection formulas in shared mem?
	// TO TRY: inner loop over z, gather ray values in shared mem

	const int X = blockIdx.x % ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockX + threadIdx.x;
	const int Y = blockIdx.x / ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockY + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.y * g_volBlockZ;
	int endZ = startZ + g_volBlockZ;
	if (endZ > dims.iVolZ)
		endZ = dims.iVolZ;

	float fX = X - 0.5f*dims.iVolX + 0.5f - 0.5f + 0.5f/dims.iRaysPerVoxelDim;
	float fY = Y - 0.5f*dims.iVolY + 0.5f - 0.5f + 0.5f/dims.iRaysPerVoxelDim;
	float fZ = startZ - 0.5f*dims.iVolZ + 0.5f - 0.5f + 0.5f/dims.iRaysPerVoxelDim;
	const float fSubStep = 1.0f/dims.iRaysPerVoxelDim;

	fOutputScale /= (dims.iRaysPerVoxelDim*dims.iRaysPerVoxelDim*dims.iRaysPerVoxelDim);


	for (int Z = startZ; Z < endZ; ++Z, fZ += 1.0f)
	{

		float fVal = 0.0f;
		float fAngle = startAngle + angleOffset + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
		{

			const float fCux = gC_C[12*angle+0];
			const float fCuy = gC_C[12*angle+1];
			const float fCuz = gC_C[12*angle+2];
			const float fCuc = gC_C[12*angle+3];
			const float fCvx = gC_C[12*angle+4];
			const float fCvy = gC_C[12*angle+5];
			const float fCvz = gC_C[12*angle+6];
			const float fCvc = gC_C[12*angle+7];
			const float fCdx = gC_C[12*angle+8];
			const float fCdy = gC_C[12*angle+9];
			const float fCdz = gC_C[12*angle+10];
			const float fCdc = gC_C[12*angle+11];

			float fXs = fX;
			for (int iSubX = 0; iSubX < dims.iRaysPerVoxelDim; ++iSubX) {
			float fYs = fY;
			for (int iSubY = 0; iSubY < dims.iRaysPerVoxelDim; ++iSubY) {
			float fZs = fZ;
			for (int iSubZ = 0; iSubZ < dims.iRaysPerVoxelDim; ++iSubZ) {

				const float fUNum = fCuc + fXs * fCux + fYs * fCuy + fZs * fCuz;
				const float fVNum = fCvc + fXs * fCvx + fYs * fCvy + fZs * fCvz;
				const float fDen = fCdc + fXs * fCdx + fYs * fCdy + fZs * fCdz;

				const float fU = fUNum / fDen;
				const float fV = fVNum / fDen;

				fVal += tex3D(gT_coneProjTexture, fU, fV, fAngle);

				fZs += fSubStep;
			}
			fYs += fSubStep;
			}
			fXs += fSubStep;
			}

		}

		volData[(Z*dims.iVolY+Y)*volPitch+X] += fVal * fOutputScale;
	}
}


bool ConeBP_Array(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SConeProjection* angles,
                  float fOutputScale)
{
	bindProjDataTexture(D_projArray);

	for (unsigned int th = 0; th < dims.iProjAngles; th += g_MaxAngles) {
		unsigned int angleCount = g_MaxAngles;
		if (th + angleCount > dims.iProjAngles)
			angleCount = dims.iProjAngles - th;

		// transfer angles to constant memory
		float* tmp = new float[12*angleCount];


		// NB: We increment angles at the end of the loop body.


#define TRANSFER_TO_CONSTANT(expr,name) do { for (unsigned int i = 0; i < angleCount; ++i) tmp[12*i+name] = (expr) ; } while (0)

		TRANSFER_TO_CONSTANT( (angles[i].fDetSZ - angles[i].fSrcZ)*angles[i].fDetVY - (angles[i].fDetSY - angles[i].fSrcY)*angles[i].fDetVZ , 0 );
		TRANSFER_TO_CONSTANT( (angles[i].fDetSX - angles[i].fSrcX)*angles[i].fDetVZ -(angles[i].fDetSZ - angles[i].fSrcZ)*angles[i].fDetVX , 1 );
		TRANSFER_TO_CONSTANT( (angles[i].fDetSY - angles[i].fSrcY)*angles[i].fDetVX - (angles[i].fDetSX - angles[i].fSrcX)*angles[i].fDetVY , 2 );
		TRANSFER_TO_CONSTANT( (angles[i].fDetSY*angles[i].fDetVZ - angles[i].fDetSZ*angles[i].fDetVY)*angles[i].fSrcX - (angles[i].fDetSX*angles[i].fDetVZ - angles[i].fDetSZ*angles[i].fDetVX)*angles[i].fSrcY + (angles[i].fDetSX*angles[i].fDetVY - angles[i].fDetSY*angles[i].fDetVX)*angles[i].fSrcZ , 3 );

		TRANSFER_TO_CONSTANT( (angles[i].fDetSY - angles[i].fSrcY)*angles[i].fDetUZ-(angles[i].fDetSZ - angles[i].fSrcZ)*angles[i].fDetUY, 4 );
		TRANSFER_TO_CONSTANT( (angles[i].fDetSZ - angles[i].fSrcZ)*angles[i].fDetUX - (angles[i].fDetSX - angles[i].fSrcX)*angles[i].fDetUZ , 5 );
		TRANSFER_TO_CONSTANT((angles[i].fDetSX - angles[i].fSrcX)*angles[i].fDetUY-(angles[i].fDetSY - angles[i].fSrcY)*angles[i].fDetUX , 6 );
		TRANSFER_TO_CONSTANT( -(angles[i].fDetSY*angles[i].fDetUZ - angles[i].fDetSZ*angles[i].fDetUY)*angles[i].fSrcX + (angles[i].fDetSX*angles[i].fDetUZ - angles[i].fDetSZ*angles[i].fDetUX)*angles[i].fSrcY - (angles[i].fDetSX*angles[i].fDetUY - angles[i].fDetSY*angles[i].fDetUX)*angles[i].fSrcZ , 7 );

		TRANSFER_TO_CONSTANT( angles[i].fDetUY*angles[i].fDetVZ - angles[i].fDetUZ*angles[i].fDetVY , 8 );
		TRANSFER_TO_CONSTANT( angles[i].fDetUZ*angles[i].fDetVX - angles[i].fDetUX*angles[i].fDetVZ , 9 );
		TRANSFER_TO_CONSTANT( angles[i].fDetUX*angles[i].fDetVY - angles[i].fDetUY*angles[i].fDetVX , 10 );
		TRANSFER_TO_CONSTANT( -angles[i].fSrcX * (angles[i].fDetUY*angles[i].fDetVZ - angles[i].fDetUZ*angles[i].fDetVY) - angles[i].fSrcY * (angles[i].fDetUZ*angles[i].fDetVX - angles[i].fDetUX*angles[i].fDetVZ) - angles[i].fSrcZ * (angles[i].fDetUX*angles[i].fDetVY - angles[i].fDetUY*angles[i].fDetVX) , 11 );

#undef TRANSFER_TO_CONSTANT
		cudaMemcpyToSymbol(gC_C, tmp, angleCount*12*sizeof(float), 0, cudaMemcpyHostToDevice); 

		delete[] tmp;

		dim3 dimBlock(g_volBlockX, g_volBlockY);

		dim3 dimGrid(((dims.iVolX/1+g_volBlockX-1)/(g_volBlockX))*((dims.iVolY/1+1*g_volBlockY-1)/(1*g_volBlockY)), (dims.iVolZ+g_volBlockZ-1)/g_volBlockZ);

		// timeval t;
		// tic(t);

		for (unsigned int i = 0; i < angleCount; i += g_anglesPerBlock) {
		// printf("Calling BP: %d, %dx%d, %dx%d to %p\n", i, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y, (void*)D_volumeData.ptr); 
			if (dims.iRaysPerVoxelDim == 1)
				dev_cone_BP<<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, fOutputScale);
			else
				dev_cone_BP_SS<<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, fOutputScale);
		}

		cudaTextForceKernelsCompletion();

		angles = angles + angleCount;
		// printf("%f\n", toc(t));

	}


	return true;
}

bool ConeBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SConeProjection* angles,
            float fOutputScale)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bool ret = ConeBP_Array(D_volumeData, cuArray, dims, angles, fOutputScale);

	cudaFreeArray(cuArray);

	return ret;
}


}

#ifdef STANDALONE
int main()
{
	astraCUDA3d::SDimensions3D dims;
	dims.iVolX = 512;
	dims.iVolY = 512;
	dims.iVolZ = 512;
	dims.iProjAngles = 496;
	dims.iProjU = 512;
	dims.iProjV = 512;
	dims.iRaysPerDetDim = 1;
	dims.iRaysPerVoxelDim = 1;

	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPitchedPtr volData; // pitch, ptr, xsize, ysize

	cudaMalloc3D(&volData, extentV);

	cudaExtent extentP;
	extentP.width = dims.iProjU*sizeof(float);
	extentP.height = dims.iProjAngles;
	extentP.depth = dims.iProjV;

	cudaPitchedPtr projData; // pitch, ptr, xsize, ysize

	cudaMalloc3D(&projData, extentP);
	cudaMemset3D(projData, 0, extentP);

#if 0
	float* slice = new float[256*256];
	cudaPitchedPtr ptr;
	ptr.ptr = slice;
	ptr.pitch = 256*sizeof(float);
	ptr.xsize = 256*sizeof(float);
	ptr.ysize = 256;

	for (unsigned int i = 0; i < 256*256; ++i)
		slice[i] = 1.0f;
	for (unsigned int i = 0; i < 256; ++i) {
		cudaExtent extentS;
		extentS.width = dims.iVolX*sizeof(float);
		extentS.height = dims.iVolY;
		extentS.depth = 1;
		cudaPos sp = { 0, 0, 0 };
		cudaPos dp = { 0, 0, i };
		cudaMemcpy3DParms p;
		p.srcArray = 0;
		p.srcPos = sp;
		p.srcPtr = ptr;
		p.dstArray = 0;
		p.dstPos = dp;
		p.dstPtr = volData;
		p.extent = extentS;
		p.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&p);
#if 0
		if (i == 128) {
			for (unsigned int j = 0; j < 256*256; ++j)
				slice[j] = 0.0f;
		}
#endif 
	}
#endif


	astraCUDA3d::SConeProjection angle[512];
	angle[0].fSrcX = -5120;
	angle[0].fSrcY = 0;
	angle[0].fSrcZ = 0;

	angle[0].fDetSX = 512;
	angle[0].fDetSY = -256;
	angle[0].fDetSZ = -256;

	angle[0].fDetUX = 0;
	angle[0].fDetUY = 1;
	angle[0].fDetUZ = 0;

	angle[0].fDetVX = 0;
	angle[0].fDetVY = 0;
	angle[0].fDetVZ = 1;

#define ROTATE0(name,i,alpha) do { angle[i].f##name##X = angle[0].f##name##X * cos(alpha) - angle[0].f##name##Y * sin(alpha); angle[i].f##name##Y = angle[0].f##name##X * sin(alpha) + angle[0].f##name##Y * cos(alpha); } while(0)
	for (int i = 1; i < 512; ++i) {
		angle[i] = angle[0];
		ROTATE0(Src, i, i*2*M_PI/512);
		ROTATE0(DetS, i, i*2*M_PI/512);
		ROTATE0(DetU, i, i*2*M_PI/512);
		ROTATE0(DetV, i, i*2*M_PI/512);
	}
#undef ROTATE0

#if 0
	astraCUDA3d::ConeFP(volData, projData, dims, angle, 1.0f);
#endif
#if 0
	float* bufs = new float[180*512];

	for (int i = 0; i < 512; ++i) {
		cudaMemcpy(bufs, ((float*)projData.ptr)+180*512*i, 180*512*sizeof(float), cudaMemcpyDeviceToHost);

		printf("%d %d %d\n", projData.pitch, projData.xsize, projData.ysize);

		char fname[20];
		sprintf(fname, "sino%03d.png", i);
		saveImage(fname, 180, 512, bufs);
	}

	float* bufp = new float[512*512];

	for (int i = 0; i < 180; ++i) {
		for (int j = 0; j < 512; ++j) {
			cudaMemcpy(bufp+512*j, ((float*)projData.ptr)+180*512*j+512*i, 512*sizeof(float), cudaMemcpyDeviceToHost);
		}

		char fname[20];
		sprintf(fname, "proj%03d.png", i);
		saveImage(fname, 512, 512, bufp);
	}
#endif		
#if 0
	for (unsigned int i = 0; i < 256*256; ++i)
		slice[i] = 0.0f;
	for (unsigned int i = 0; i < 256; ++i) {
		cudaExtent extentS;
		extentS.width = dims.iVolX*sizeof(float);
		extentS.height = dims.iVolY;
		extentS.depth = 1;
		cudaPos sp = { 0, 0, 0 };
		cudaPos dp = { 0, 0, i };
		cudaMemcpy3DParms p;
		p.srcArray = 0;
		p.srcPos = sp;
		p.srcPtr = ptr;
		p.dstArray = 0;
		p.dstPos = dp;
		p.dstPtr = volData;
		p.extent = extentS;
		p.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&p);
	}
#endif

	astraCUDA3d::ConeBP(volData, projData, dims, angle, 1.0f);
#if 0
	float* buf = new float[256*256];

	for (int i = 0; i < 256; ++i) {
		cudaMemcpy(buf, ((float*)volData.ptr)+256*256*i, 256*256*sizeof(float), cudaMemcpyDeviceToHost);

		printf("%d %d %d\n", volData.pitch, volData.xsize, volData.ysize);

		char fname[20];
		sprintf(fname, "vol%03d.png", i);
		saveImage(fname, 256, 256, buf);
	}
#endif

}
#endif
