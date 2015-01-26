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

#ifndef _INC_ASTRA_FOURIER
#define _INC_ASTRA_FOURIER

#include "Globals.h"

namespace astra {


/**
 * Perform a 1D DFT or inverse DFT.
 *
 * @param iLength number of elements
 * @param pfRealIn real part of input
 * @param pfImaginaryIn imaginary part of input
 * @param pfRealOut real part of output
 * @param pfImaginaryOut imaginary part of output
 * @param iStrideIn distance between elements in pf*In
 * @param iStrideOut distance between elements in pf*Out
 * @param bInverse if true, perform an inverse DFT
 */

void _AstraExport discreteFourierTransform1D(unsigned int iLength,
                                const float32* pfRealIn,
                                const float32* pfImaginaryIn,
                                float32* pfRealOut,
                                float32* pfImaginaryOut,
                                unsigned int iStrideIn,
                                unsigned int iStrideOut,
                                bool bInverse);

/**
 * Perform a 2D DFT or inverse DFT.
 *
 * @param iHeight number of rows
 * @param iWidth number of columns
 * @param pfRealIn real part of input
 * @param pfImaginaryIn imaginary part of input
 * @param pfRealOut real part of output
 * @param pfImaginaryOut imaginary part of output
 * @param bInverse if true, perform an inverse DFT
 */

void _AstraExport discreteFourierTransform2D(unsigned int iHeight, unsigned int iWidth,
                                const float32* pfRealIn,
                                const float32* pfImaginaryIn,
                                float32* pfRealOut,
                                float32* pfImaginaryOut,
                                bool bInverse);

/**
 * Perform a 1D FFT or inverse FFT. The size must be a power of two.
 * This transform can be done in-place, so the input and output pointers
 * may point to the same data.
 *
 * @param iLength number of elements, must be a power of two
 * @param pfRealIn real part of input
 * @param pfImaginaryIn imaginary part of input
 * @param pfRealOut real part of output
 * @param pfImaginaryOut imaginary part of output
 * @param iStrideIn distance between elements in pf*In
 * @param iStrideOut distance between elements in pf*Out
 * @param bInverse if true, perform an inverse DFT
 */

void _AstraExport fastTwoPowerFourierTransform1D(unsigned int iLength,
                                    const float32* pfRealIn,
                                    const float32* pfImaginaryIn,
                                    float32* pfRealOut,
                                    float32* pfImaginaryOut,
                                    unsigned int iStrideIn,
                                    unsigned int iStrideOut,
                                    bool bInverse);

/**
 * Perform a 2D FFT or inverse FFT. The size must be a power of two.
 * This transform can be done in-place, so the input and output pointers
 * may point to the same data.
 *
 * @param iHeight number of rows, must be a power of two
 * @param iWidth number of columns, must be a power of two
 * @param pfRealIn real part of input
 * @param pfImaginaryIn imaginary part of input
 * @param pfRealOut real part of output
 * @param pfImaginaryOut imaginary part of output
 * @param bInverse if true, perform an inverse DFT
 */

void _AstraExport fastTwoPowerFourierTransform2D(unsigned int iHeight,
                                    unsigned int iWidth,
                                    const float32* pfRealIn,
                                    const float32* pfImaginaryIn,
                                    float32* pfRealOut,
                                    float32* pfImaginaryOut,
                                    bool bInverse);


}

#endif
