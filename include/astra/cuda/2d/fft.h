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

#ifndef _CUDA_FFT_H
#define _CUDA_FFT_H

#include "astra/cuda/gpu_fft_wrapper.h"

#include <optional>

#include "astra/Filters.h"

namespace astraCUDA {

// Functions taking an std::optional<cudaStream_t> will be
// synchronous when not passed a stream. If they do get a stream,
// the cuda parts might be partially or fully asynchronous.

bool allocateComplexOnDevice(int _iProjectionCount,
                             int _iDetectorCount,
                             cufftComplex ** _ppDevComplex);

bool freeComplexOnDevice(cufftComplex * _pDevComplex);

bool uploadComplexArrayToDevice(int _iProjectionCount, int _iDetectorCount,
                                cufftComplex * _pHostComplexSource,
                                cufftComplex * _pDevComplexTarget,
                                std::optional<cudaStream_t> _stream = {});

bool runCudaFFT(int _iProjectionCount, const float * D_pfSource,
                int _iSourcePitch, int _iProjDets,
                int _iPaddedSize,
                cufftComplex * D_pcTarget,
                std::optional<cudaStream_t> _stream = {});

bool runCudaIFFT(int _iProjectionCount, const cufftComplex* D_pcSource,
                 float * D_pfTarget,
                 int _iTargetPitch, int _iProjDets,
                 int _iPaddedSize,
                 std::optional<cudaStream_t> _stream = {});

bool applyFilter(int _iProjectionCount, int _iFreqBinCount,
                 cufftComplex * _pSinogram, cufftComplex * _pFilter,
                 bool singleFilter = false,
                 std::optional<cudaStream_t> _stream = {});

void genCuFFTFilter(const astra::SFilterConfig &_cfg, int _iProjectionCount,
                   cufftComplex * _pFilter, int _iFFTRealDetectorCount,
                   int _iFFTFourierDetectorCount);

bool prepareCuFFTFilter(const astra::SFilterConfig &cfg,
                        cufftComplex *&D_filter,
                        bool &singleFilter,
                        int iProjectionCount, int iDetectorCount,
                        std::optional<cudaStream_t> _stream = {});

void genIdenFilter(int _iProjectionCount, cufftComplex * _pFilter,
                   int _iFFTRealDetectorCount, int _iFFTFourierDetectorCount);

bool rescaleInverseFourier(int _iProjectionCount, int _iDetectorCount,
                           float * _pfInFourierOutput,
                           std::optional<cudaStream_t> _stream = {});

}

#endif /* FFT_H */
