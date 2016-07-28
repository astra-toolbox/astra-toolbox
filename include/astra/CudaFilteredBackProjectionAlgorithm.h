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

#ifndef CUDAFILTEREDBACKPROJECTIONALGORITHM2_H
#define CUDAFILTEREDBACKPROJECTIONALGORITHM2_H

#include <astra/Float32ProjectionData2D.h>
#include <astra/Float32VolumeData2D.h>
#include <astra/ReconstructionAlgorithm2D.h>

#include "../../cuda/2d/astra.h"

namespace astra
{

class _AstraExport CCudaFilteredBackProjectionAlgorithm : public CReconstructionAlgorithm2D
{
public:
	static std::string type;

private:
	CFloat32ProjectionData2D * m_pSinogram;
	CFloat32VolumeData2D * m_pReconstruction;
	int m_iGPUIndex;
	int m_iPixelSuperSampling;
	E_FBPFILTER m_eFilter;
	float * m_pfFilter;
	int m_iFilterWidth;	// number of elements per projection direction in filter
	float m_fFilterParameter;  // some filters allow for parameterization (value < 0.0f -> no parameter)
	float m_fFilterD;	// frequency cut-off
	bool m_bShortScan; // short-scan mode for fan beam

	static E_FBPFILTER _convertStringToFilter(const char * _filterType);

public:
	CCudaFilteredBackProjectionAlgorithm();
	virtual ~CCudaFilteredBackProjectionAlgorithm();

	virtual bool initialize(const Config& _cfg);
	bool initialize(CFloat32ProjectionData2D * _pSinogram, CFloat32VolumeData2D * _pReconstruction, E_FBPFILTER _eFilter, const float * _pfFilter = NULL, int _iFilterWidth = 0, int _iGPUIndex = -1, float _fFilterParameter = -1.0f);

	virtual void run(int _iNrIterations = 0);

	static int calcIdealRealFilterWidth(int _iDetectorCount);
	static int calcIdealFourierFilterWidth(int _iDetectorCount);
	
	//debug
	static void testGenFilter(E_FBPFILTER _eFilter, float _fD, int _iProjectionCount, cufftComplex * _pFilter, int _iFFTRealDetectorCount, int _iFFTFourierDetectorCount);
	static int getGPUCount();

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

protected:
	bool check();

	AstraFBP* m_pFBP;

	bool m_bAstraFBPInit;

	void initializeFromProjector();
	virtual bool requiresProjector() const { return false; }
};

// inline functions
inline std::string CCudaFilteredBackProjectionAlgorithm::description() const { return CCudaFilteredBackProjectionAlgorithm::type; };

}

#endif /* CUDAFILTEREDBACKPROJECTIONALGORITHM2_H */
