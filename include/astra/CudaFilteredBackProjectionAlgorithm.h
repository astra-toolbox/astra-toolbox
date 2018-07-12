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

#ifndef CUDAFILTEREDBACKPROJECTIONALGORITHM_H
#define CUDAFILTEREDBACKPROJECTIONALGORITHM_H

#ifdef ASTRA_CUDA

#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"
#include "CudaReconstructionAlgorithm2D.h"
#include "Filters.h"

#include "cuda/2d/astra.h"

namespace astra
{

class _AstraExport CCudaFilteredBackProjectionAlgorithm : public CCudaReconstructionAlgorithm2D
{
public:
	static std::string type;

private:
	SFilterConfig m_filterConfig;
	bool m_bShortScan; // short-scan mode for fan beam

public:
	CCudaFilteredBackProjectionAlgorithm();
	virtual ~CCudaFilteredBackProjectionAlgorithm();

	virtual bool initialize(const Config& _cfg);
	bool initialize(CFloat32ProjectionData2D * _pSinogram, CFloat32VolumeData2D * _pReconstruction, E_FBPFILTER _eFilter, const float * _pfFilter = NULL, int _iFilterWidth = 0, int _iGPUIndex = -1, float _fFilterParameter = -1.0f);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

protected:
	bool check();

	virtual void initCUDAAlgorithm();
};

// inline functions
inline std::string CCudaFilteredBackProjectionAlgorithm::description() const { return CCudaFilteredBackProjectionAlgorithm::type; };

}

#endif

#endif /* CUDAFILTEREDBACKPROJECTIONALGORITHM2_H */
