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

#include "astra/Fourier.h"

namespace astra {


void discreteFourierTransform1D(unsigned int iLength,
                                const float32* pfRealIn,
                                const float32* pfImaginaryIn,
                                float32* pfRealOut,
                                float32* pfImaginaryOut,
                                unsigned int iStrideIn,
                                unsigned int iStrideOut,
                                bool inverse)
{
	for (unsigned int w = 0; w < iLength; w++)
	{
		pfRealOut[iStrideOut*w] = pfImaginaryOut[iStrideOut*w] = 0;
		for (unsigned int y = 0; y < iLength; y++)
		{
			float32 a = 2 * PI * w * y / float32(iLength);
			if (!inverse)
				a = -a;
			float32 ca = cos(a);
			float32 sa = sin(a);
			pfRealOut[iStrideOut*w] += pfRealIn[iStrideIn*y] * ca - pfImaginaryIn[iStrideIn*y] * sa;
			pfImaginaryOut[iStrideOut*w] += pfRealIn[iStrideIn*y] * sa + pfImaginaryIn[iStrideIn*y] * ca;   
		}
	}

	if (inverse) {
		for (unsigned int x = 0; x < iLength; ++x) {
			pfRealOut[iStrideOut*x] /= iLength;
			pfImaginaryOut[iStrideOut*x] /= iLength;
		}
	}
}

void discreteFourierTransform2D(unsigned int iHeight, unsigned int iWidth,
                                const float32* pfRealIn,
                                const float32* pfImaginaryIn,
                                float32* pfRealOut,
                                float32* pfImaginaryOut,
                                bool inverse)
{
	float32* reTemp = new float32[iWidth * iHeight];
	float32* imTemp = new float32[iWidth * iHeight];

	//calculate the fourier transform of the columns
	for (unsigned int x = 0; x < iWidth; x++)
	{
		discreteFourierTransform1D(iHeight, pfRealIn+x, pfImaginaryIn+x,
		                           reTemp+x, imTemp+x,
		                           iWidth, iWidth, inverse);
	}

	//calculate the fourier transform of the rows
	for(unsigned int y = 0; y < iHeight; y++)
	{
		discreteFourierTransform1D(iWidth,
		                           reTemp+y*iWidth,
		                           imTemp+y*iWidth,
		                           pfRealOut+y*iWidth,
		                           pfImaginaryOut+y*iWidth,
		                           1, 1, inverse);
	}

	delete[] reTemp;
	delete[] imTemp;
}

/** permute the entries from pfDataIn into pfDataOut to prepare for an
 *  in-place FFT. pfDataIn may be equal to pfDataOut.
 */
static void bitReverse(unsigned int iLength,
                       const float32* pfDataIn, float32* pfDataOut,
                       unsigned int iStrideShiftIn,
                       unsigned int iStrideShiftOut)
{
	if (pfDataIn == pfDataOut) {
		assert(iStrideShiftIn == iStrideShiftOut);
		float32 t;
		unsigned int j = 0;
		for(unsigned int i = 0; i < iLength - 1; i++) {
			if (i < j) {
				t = pfDataOut[i<<iStrideShiftOut];
				pfDataOut[i<<iStrideShiftOut] = pfDataOut[j<<iStrideShiftOut];
				pfDataOut[j<<iStrideShiftOut] = t;
			}
			unsigned int k = iLength / 2;
			while (k <= j) {
				j -= k;
				k /= 2;
			}
			j += k;
		}
	} else {
		unsigned int j = 0;
		for(unsigned int i = 0; i < iLength - 1; i++) {
			pfDataOut[i<<iStrideShiftOut] = pfDataIn[j<<iStrideShiftIn];
			unsigned int k = iLength / 2;
			while (k <= j) {
				j -= k;
				k /= 2;
			}
			j += k;
		}
		pfDataOut[(iLength-1)<<iStrideShiftOut] = pfDataIn[(iLength-1)<<iStrideShiftOut];
	}
}

static unsigned int log2(unsigned int n)
{
	unsigned int l = 0;
	while (n > 1) {
		n /= 2;
		++l;
	}
	return l;
}

/** perform 1D FFT. iLength, iStrideIn, iStrideOut must be powers of two. */
void fastTwoPowerFourierTransform1D(unsigned int iLength,
                                    const float32* pfRealIn,
                                    const float32* pfImaginaryIn,
                                    float32* pfRealOut,
                                    float32* pfImaginaryOut,
                                    unsigned int iStrideIn,
                                    unsigned int iStrideOut,
                                    bool inverse)
{
	unsigned int iStrideShiftIn = log2(iStrideIn);
	unsigned int iStrideShiftOut = log2(iStrideOut);
	unsigned int iLogLength = log2(iLength);

	bitReverse(iLength, pfRealIn, pfRealOut, iStrideShiftIn, iStrideShiftOut);
	bitReverse(iLength, pfImaginaryIn, pfImaginaryOut, iStrideShiftIn, iStrideShiftOut);

	float32 ca = -1.0;
	float32 sa = 0.0;
	unsigned int l1 = 1, l2 = 1;
	for(unsigned int l=0; l < iLogLength; ++l)
	{
		l1 = l2;
		l2 *= 2;
		float32 u1 = 1.0;
		float32 u2 = 0.0;
		for(unsigned int j = 0; j < l1; j++)
		{
			for(unsigned int i = j; i < iLength; i += l2)
			{
				unsigned int i1 = i + l1;
				float32 t1 = u1 * pfRealOut[i1<<iStrideShiftOut] - u2 * pfImaginaryOut[i1<<iStrideShiftOut];
				float32 t2 = u1 * pfImaginaryOut[i1<<iStrideShiftOut] + u2 * pfRealOut[i1<<iStrideShiftOut];
				pfRealOut[i1<<iStrideShiftOut] = pfRealOut[i<<iStrideShiftOut] - t1;
				pfImaginaryOut[i1<<iStrideShiftOut] = pfImaginaryOut[i<<iStrideShiftOut] - t2;
				pfRealOut[i<<iStrideShiftOut] += t1;
				pfImaginaryOut[i<<iStrideShiftOut] += t2;
			}
			float32 z =  u1 * ca - u2 * sa;
			u2 = u1 * sa + u2 * ca;
			u1 = z;
		}
		sa = sqrt((1.0 - ca) / 2.0);
		if (!inverse) 
			sa = -sa;
		ca = sqrt((1.0 + ca) / 2.0);
	}

	if (inverse) {
		for (unsigned int i = 0; i < iLength; ++i) {
			pfRealOut[i<<iStrideShiftOut] /= iLength;
			pfImaginaryOut[i<<iStrideShiftOut] /= iLength;
		}
	}
}

void fastTwoPowerFourierTransform2D(unsigned int iHeight,
                                    unsigned int iWidth,
                                    const float32* pfRealIn,
                                    const float32* pfImaginaryIn,
                                    float32* pfRealOut,
                                    float32* pfImaginaryOut,
                                    bool inverse)
{
	//calculate the fourier transform of the columns
	for (unsigned int x = 0; x < iWidth; x++)
	{
		fastTwoPowerFourierTransform1D(iHeight, pfRealIn+x, pfImaginaryIn+x,
		                               pfRealOut+x, pfImaginaryOut+x,
		                               iWidth, iWidth, inverse);
	}

	//calculate the fourier transform of the rows
	for (unsigned int y = 0; y < iHeight; y++)
	{
		fastTwoPowerFourierTransform1D(iWidth,
		                               pfRealOut+y*iWidth,
		                               pfImaginaryOut+y*iWidth,
		                               pfRealOut+y*iWidth,
		                               pfImaginaryOut+y*iWidth,
		                               1, 1, inverse);
	}
}

}
