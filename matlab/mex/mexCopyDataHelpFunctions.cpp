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

#include "mexCopyDataHelpFunctions.h"

#include "mexHelpFunctions.h"

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__SSE2__)
# include <emmintrin.h>
# define STORE_32F_64F_CORE8(in, out, count, base) \
		{\
			const __m128 inV0 = *((const __m128 *) &in[count + 0 + base]);\
			const __m128 inV1 = *((const __m128 *) &in[count + 4 + base]);\
\
			*((__m128d *)&out[count + 0 + base]) = _mm_cvtps_pd(inV0);\
			*((__m128d *)&out[count + 2 + base]) = _mm_cvtps_pd(_mm_movehl_ps(inV0, inV0));\
\
			*((__m128d *)&out[count + 4 + base]) = _mm_cvtps_pd(inV1);\
			*((__m128d *)&out[count + 6 + base]) = _mm_cvtps_pd(_mm_movehl_ps(inV1, inV1));\
		}
# define STORE_64F_32F_CORE8(in, out, count, base) \
		{\
			const __m128d inV0 = *((const __m128d *) &in[count + 0 + base]);\
			const __m128d inV1 = *((const __m128d *) &in[count + 2 + base]);\
\
			const __m128d inV2 = *((const __m128d *) &in[count + 4 + base]);\
			const __m128d inV3 = *((const __m128d *) &in[count + 6 + base]);\
\
			*((__m128 *)&out[count + 0 + base]) = _mm_movelh_ps(_mm_cvtpd_ps(inV0), _mm_cvtpd_ps(inV1));\
			*((__m128 *)&out[count + 4 + base]) = _mm_movelh_ps(_mm_cvtpd_ps(inV2), _mm_cvtpd_ps(inV3));\
		}
# define STORE_32F_32F_CORE8(in, out, count, base) \
		{\
			*((__m128 *)&out[count + 0 + base]) = *((const __m128 *)&in[count + 0 + base]);\
			*((__m128 *)&out[count + 4 + base]) = *((const __m128 *)&in[count + 4 + base]);\
		}
# define STORE_CONST_32F_CORE8(val, out, count, base) \
		{\
			*((__m128 *)&out[count + 0 + base]) = val;\
			*((__m128 *)&out[count + 4 + base]) = val;\
		}
#else
# define STORE_32F_64F_CORE8(in, out, count, base) \
		{\
			out[count + 0 + base] = (double)in[count + 0 + base];\
			out[count + 1 + base] = (double)in[count + 1 + base];\
			out[count + 2 + base] = (double)in[count + 2 + base];\
			out[count + 3 + base] = (double)in[count + 3 + base];\
			out[count + 4 + base] = (double)in[count + 4 + base];\
			out[count + 5 + base] = (double)in[count + 5 + base];\
			out[count + 6 + base] = (double)in[count + 6 + base];\
			out[count + 7 + base] = (double)in[count + 7 + base];\
		}
# define STORE_64F_32F_CORE8(in, out, count, base) \
		{\
			out[count + 0 + base] = (float)in[count + 0 + base];\
			out[count + 1 + base] = (float)in[count + 1 + base];\
			out[count + 2 + base] = (float)in[count + 2 + base];\
			out[count + 3 + base] = (float)in[count + 3 + base];\
			out[count + 4 + base] = (float)in[count + 4 + base];\
			out[count + 5 + base] = (float)in[count + 5 + base];\
			out[count + 6 + base] = (float)in[count + 6 + base];\
			out[count + 7 + base] = (float)in[count + 7 + base];\
		}
# define STORE_32F_32F_CORE8(in, out, count, base) \
		{\
			out[count + 0 + base] = in[count + 0 + base];\
			out[count + 1 + base] = in[count + 1 + base];\
			out[count + 2 + base] = in[count + 2 + base];\
			out[count + 3 + base] = in[count + 3 + base];\
			out[count + 4 + base] = in[count + 4 + base];\
			out[count + 5 + base] = in[count + 5 + base];\
			out[count + 6 + base] = in[count + 6 + base];\
			out[count + 7 + base] = in[count + 7 + base];\
		}
#endif
#define STORE_8F_32F_CORE8(in, out, count, base) \
		{\
			out[count + 0 + base] = (float)in[count + 0 + base];\
			out[count + 1 + base] = (float)in[count + 1 + base];\
			out[count + 2 + base] = (float)in[count + 2 + base];\
			out[count + 3 + base] = (float)in[count + 3 + base];\
			out[count + 4 + base] = (float)in[count + 4 + base];\
			out[count + 5 + base] = (float)in[count + 5 + base];\
			out[count + 6 + base] = (float)in[count + 6 + base];\
			out[count + 7 + base] = (float)in[count + 7 + base];\
		}

const char * warnDataTypeNotSupported = "Data type not supported: nothing was copied";

void
_copyMexToCFloat32Array(const mxArray * const inArray, astra::float32 * const out)
{
	const long long tot_size = mxGetNumberOfElements(inArray);
	const long long block = 32;
	const long long totRoundedPixels = ROUND_DOWN(tot_size, block);

	// Array of doubles
	if (mxIsDouble(inArray)) {
		const double * const pdMatlabData = mxGetPr(inArray);

#pragma omp for nowait
		for (long long count = 0; count < totRoundedPixels; count += block) {
			STORE_64F_32F_CORE8(pdMatlabData, out, count,  0);
			STORE_64F_32F_CORE8(pdMatlabData, out, count,  8);
			STORE_64F_32F_CORE8(pdMatlabData, out, count, 16);
			STORE_64F_32F_CORE8(pdMatlabData, out, count, 24);
		}
#pragma omp for nowait
		for (long long count = totRoundedPixels; count < tot_size; count++) {
			out[count] = pdMatlabData[count];
		}
	}

	// Array of floats
	else if (mxIsSingle(inArray)) {
		const float * const pfMatlabData = (const float *)mxGetData(inArray);

#pragma omp for nowait
		for (long long count = 0; count < totRoundedPixels; count += block) {
			STORE_32F_32F_CORE8(pfMatlabData, out, count,  0);
			STORE_32F_32F_CORE8(pfMatlabData, out, count,  8);
			STORE_32F_32F_CORE8(pfMatlabData, out, count, 16);
			STORE_32F_32F_CORE8(pfMatlabData, out, count, 24);
		}
#pragma omp for nowait
		for (long long count = totRoundedPixels; count < tot_size; count++) {
			out[count] = pfMatlabData[count];
		}
	}

	// Array of logicals
	else if (mxIsLogical(inArray)) {
		const mxLogical * const pfMatlabData = (const mxLogical *)mxGetLogicals(inArray);

#pragma omp for nowait
		for (long long count = 0; count < totRoundedPixels; count += block) {
			STORE_8F_32F_CORE8(pfMatlabData, out, count,  0);
			STORE_8F_32F_CORE8(pfMatlabData, out, count,  8);
			STORE_8F_32F_CORE8(pfMatlabData, out, count, 16);
			STORE_8F_32F_CORE8(pfMatlabData, out, count, 24);
		}
#pragma omp for nowait
		for (long long count = totRoundedPixels; count < tot_size; count++) {
			out[count] = pfMatlabData[count];
		}
	}
	else {
#pragma omp single nowait
		mexWarnMsgIdAndTxt("ASTRA_MEX:wrong_datatype", warnDataTypeNotSupported);
	}
}

void
_copyCFloat32ArrayToMex(const astra::float32 * const in, mxArray * const outArray)
{
	const long long tot_size = mxGetNumberOfElements(outArray);
	const long long block = 32;
	const long long tot_rounded_size = ROUND_DOWN(tot_size, block);

	if (mxIsDouble(outArray)) {
		double * const pdMatlabData = mxGetPr(outArray);

#pragma omp for nowait
		for (long long count = 0; count < tot_rounded_size; count += block) {
			STORE_32F_64F_CORE8(in, pdMatlabData, count,  0);
			STORE_32F_64F_CORE8(in, pdMatlabData, count,  8);
			STORE_32F_64F_CORE8(in, pdMatlabData, count, 16);
			STORE_32F_64F_CORE8(in, pdMatlabData, count, 24);
		}
#pragma omp for nowait
		for (long long count = tot_rounded_size; count < tot_size; count++) {
			pdMatlabData[count] = in[count];
		}
	}
	else if (mxIsSingle(outArray)) {
		float * const pfMatlabData = (float *) mxGetData(outArray);

#pragma omp for nowait
		for (long long count = 0; count < tot_rounded_size; count += block) {
			STORE_32F_32F_CORE8(in, pfMatlabData, count,  0);
			STORE_32F_32F_CORE8(in, pfMatlabData, count,  8);
			STORE_32F_32F_CORE8(in, pfMatlabData, count, 16);
			STORE_32F_32F_CORE8(in, pfMatlabData, count, 24);
		}
#pragma omp for nowait
		for (long long count = tot_rounded_size; count < tot_size; count++) {
			pfMatlabData[count] = in[count];
		}
	}
	else {
#pragma omp single nowait
		mexWarnMsgIdAndTxt("ASTRA_MEX:wrong_datatype", warnDataTypeNotSupported);
	}
}

void
_initializeCFloat32Array(const astra::float32 & val, astra::float32 * const out,
		const size_t & tot_size)
{
#ifdef __SSE2__
	const long long block = 32;
	const long long tot_rounded_size = ROUND_DOWN(tot_size, block);

	const __m128 vecVal = _mm_set1_ps(val);

	{
#pragma omp for nowait
		for (long long count = 0; count < tot_rounded_size; count += block) {
			STORE_CONST_32F_CORE8(vecVal, out, count,  0);
			STORE_CONST_32F_CORE8(vecVal, out, count,  8);
			STORE_CONST_32F_CORE8(vecVal, out, count, 16);
			STORE_CONST_32F_CORE8(vecVal, out, count, 24);
		}
#else
	const long long tot_rounded_size = 0;
	{
#endif
#pragma omp for nowait
		for (long long count = tot_rounded_size; count < (long long)tot_size; count++) {
			out[count] = val;
		}
	}
}

void
copyMexToCFloat32Array(const mxArray * const in,
		astra::float32 * const out, const size_t &tot_size)
{
#pragma omp parallel
	{
		// fill with scalar value
		if (mexIsScalar(in) || mxIsEmpty(in)) {
			astra::float32 fValue = 0.f;
			if (!mxIsEmpty(in)) {
				fValue = (astra::float32)mxGetScalar(in);
			}
			_initializeCFloat32Array(fValue, out, tot_size);
		}
		// fill with array value
		else {
			_copyMexToCFloat32Array(in, out);
		}
	}
}

void
copyCFloat32ArrayToMex(const float * const in, mxArray * const outArray)
{
#pragma omp parallel
	{
		_copyCFloat32ArrayToMex(in, outArray);
	}
}
