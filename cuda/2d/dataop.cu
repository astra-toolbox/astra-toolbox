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

#include "util.h"
#include "dataop.h"
#include "arith.h"
#include <cassert>

namespace astraCUDA {

void operationVolumeMult(float* data1, float* data2, unsigned int width, unsigned int height)
{
	float* D_data1;
	float* D_data2;

	unsigned int pitch;
	allocateVolume(D_data1, width, height, pitch);
	copyVolumeToDevice(data1, width, width, height, D_data1, pitch);

	allocateVolume(D_data2, width, height, pitch);
	copyVolumeToDevice(data2, width, width, height, D_data2, pitch);

	processVol<opMul, VOL>(D_data1, D_data2, pitch, width, height);

	copyVolumeFromDevice(data1, width, width, height, D_data1, pitch);

	cudaFree(D_data1);
	cudaFree(D_data2);
}

void operationVolumeMultScalarMask(float* data, float* mask, float scalar, unsigned int width, unsigned int height)
{
	float* D_data;
	float* D_mask;

	unsigned int pitch;
	allocateVolume(D_data, width, height, pitch);
	copyVolumeToDevice(data, width, width, height, D_data, pitch);

	allocateVolume(D_mask, width, height, pitch);
	copyVolumeToDevice(mask, width, width, height, D_mask, pitch);

	processVol<opMulMask, VOL>(D_data, D_mask, scalar, pitch, width, height);

	copyVolumeFromDevice(data, width, width, height, D_data, pitch);

	cudaFree(D_data);
	cudaFree(D_mask);
}


void operationVolumeMultScalar(float* data, float scalar, unsigned int width, unsigned int height)
{
	float* D_data;

	unsigned int pitch;
	allocateVolume(D_data, width, height, pitch);
	copyVolumeToDevice(data, width, width, height, D_data, pitch);

	processVol<opMul, VOL>(D_data, scalar, pitch, width, height);

	copyVolumeFromDevice(data, width, width, height, D_data, pitch);

	cudaFree(D_data);
}


void operationVolumeAdd(float* data1, float* data2, unsigned int width, unsigned int height)
{
	float* D_data1;
	float* D_data2;

	unsigned int pitch;
	allocateVolume(D_data1, width, height, pitch);
	copyVolumeToDevice(data1, width, width, height, D_data1, pitch);

	allocateVolume(D_data2, width, height, pitch);
	copyVolumeToDevice(data2, width, width, height, D_data2, pitch);

	processVol<opAdd, VOL>(D_data1, D_data2, pitch, width, height);

	copyVolumeFromDevice(data1, width, width, height, D_data1, pitch);

	cudaFree(D_data1);
	cudaFree(D_data2);
}


void operationVolumeAddScalar(float* data, float scalar, unsigned int width, unsigned int height)
{
	float* D_data;

	unsigned int pitch;
	allocateVolume(D_data, width, height, pitch);
	copyVolumeToDevice(data, width, width, height, D_data, pitch);

	processVol<opAdd, VOL>(D_data, scalar, pitch, width, height);

	copyVolumeFromDevice(data, width, width, height, D_data, pitch);

	cudaFree(D_data);
}


}
