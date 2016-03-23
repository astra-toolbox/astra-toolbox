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

#ifndef _INC_ASTRA_DATAPROJECTORPOLICIES
#define _INC_ASTRA_DATAPROJECTORPOLICIES

#include "Globals.h"
#include "Config.h"

#include <list>

#include "Float32ProjectionData2D.h"
#include "Float32VolumeData2D.h"

namespace astra {

//enum {PixelDrivenPolicy, RayDrivenPolicy, AllPolicy} PolicyType;


//----------------------------------------------------------------------------------------
/** Policy for Default Forward Projection (Ray Driven)
 */
class DefaultFPPolicy {

	//< Projection Data
	CFloat32ProjectionData2D* m_pProjectionData;
	//< Volume Data
	CFloat32VolumeData2D* m_pVolumeData;

public:
	FORCEINLINE DefaultFPPolicy();
	FORCEINLINE DefaultFPPolicy(CFloat32VolumeData2D* _pVolumeData, CFloat32ProjectionData2D* _pProjectionData);
	FORCEINLINE ~DefaultFPPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};


//----------------------------------------------------------------------------------------
/** Policy for Default Back Projection. (Ray+Pixel Driven)
 *  This does VolumeData += transpose(ProjectionMap) * ProjectionData.
 */
class DefaultBPPolicy {

	//< Projection Data
	CFloat32ProjectionData2D* m_pProjectionData;
	//< Volume Data
	CFloat32VolumeData2D* m_pVolumeData;

public:
	FORCEINLINE DefaultBPPolicy();
	FORCEINLINE DefaultBPPolicy(CFloat32VolumeData2D* _pVolumeData, CFloat32ProjectionData2D* _pProjectionData);
	FORCEINLINE ~DefaultBPPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};



//----------------------------------------------------------------------------------------
/** Policy For Calculating the Projection Difference between Volume Data and Projection Data (Ray Driven)
 */
class DiffFPPolicy {

	CFloat32ProjectionData2D* m_pDiffProjectionData;
	CFloat32ProjectionData2D* m_pBaseProjectionData;
	CFloat32VolumeData2D* m_pVolumeData;
public:

	FORCEINLINE DiffFPPolicy();
	FORCEINLINE DiffFPPolicy(CFloat32VolumeData2D* _vol_data, CFloat32ProjectionData2D* _proj_data, CFloat32ProjectionData2D* _proj_data_base);
	FORCEINLINE ~DiffFPPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Store Pixel Weights (Ray+Pixel Driven)
 */
class StorePixelWeightsPolicy {

	SPixelWeight* m_pPixelWeights;
	int m_iMaxPixelCount;
	int m_iStoredPixelCount;

public:

	FORCEINLINE StorePixelWeightsPolicy();
	FORCEINLINE StorePixelWeightsPolicy(SPixelWeight* _pPixelWeights, int _iMaxPixelCount);
	FORCEINLINE ~StorePixelWeightsPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);

	FORCEINLINE int getStoredPixelCount();
};


//----------------------------------------------------------------------------------------
/** Policy For Calculating the Total Pixel Weight Multiplied by Sinogram
 */
class TotalPixelWeightBySinogramPolicy {

	CFloat32VolumeData2D* m_pPixelWeight;
	CFloat32ProjectionData2D* m_pSinogram;

public:

	FORCEINLINE TotalPixelWeightBySinogramPolicy();
	FORCEINLINE TotalPixelWeightBySinogramPolicy(CFloat32ProjectionData2D* _pSinogram, CFloat32VolumeData2D* _pPixelWeight);
	FORCEINLINE ~TotalPixelWeightBySinogramPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For Calculating the Total Pixel Weight
 */
class TotalPixelWeightPolicy {

	CFloat32VolumeData2D* m_pPixelWeight;

public:

	FORCEINLINE TotalPixelWeightPolicy();
	FORCEINLINE TotalPixelWeightPolicy(CFloat32VolumeData2D* _pPixelWeight);
	FORCEINLINE ~TotalPixelWeightPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For Calculating the the Total Ray Length
 */
class TotalRayLengthPolicy {

	CFloat32ProjectionData2D* m_pRayLength;

public:

	FORCEINLINE TotalRayLengthPolicy();
	FORCEINLINE TotalRayLengthPolicy(CFloat32ProjectionData2D* _pRayLength);
	FORCEINLINE ~TotalRayLengthPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};


//----------------------------------------------------------------------------------------
/** Policy For Combining Two Policies
 */
template<typename P1, typename P2>
class CombinePolicy {

	P1 policy1;
	P2 policy2;

public:

	FORCEINLINE CombinePolicy();
	FORCEINLINE CombinePolicy(P1 _policy1, P2 _policy2);
	FORCEINLINE ~CombinePolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For Combining Three Policies
 */
template<typename P1, typename P2, typename P3>
class Combine3Policy {

	P1 policy1;
	P2 policy2;
	P3 policy3;

public:

	FORCEINLINE Combine3Policy();
	FORCEINLINE Combine3Policy(P1 _policy1, P2 _policy2, P3 _policy3);
	FORCEINLINE ~Combine3Policy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For Combining Four Policies
 */
template<typename P1, typename P2, typename P3, typename P4>
class Combine4Policy {

	P1 policy1;
	P2 policy2;
	P3 policy3;
	P4 policy4;

public:

	FORCEINLINE Combine4Policy();
	FORCEINLINE Combine4Policy(P1 _policy1, P2 _policy2, P3 _policy3, P4 _policy4);
	FORCEINLINE ~Combine4Policy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For Combining a List of the same Policies
 */
template<typename P>
class CombineListPolicy {

	std::vector<P> policyList;
	unsigned int size;

public:

	FORCEINLINE CombineListPolicy();
	FORCEINLINE CombineListPolicy(std::vector<P> _policyList);
	FORCEINLINE ~CombineListPolicy();

	FORCEINLINE void addPolicy(P _policy);

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Empty Policy
 */
class EmptyPolicy {

public:

	FORCEINLINE EmptyPolicy();
	FORCEINLINE ~EmptyPolicy();	

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For SIRT Backprojection
 */
class SIRTBPPolicy {

	CFloat32ProjectionData2D* m_pSinogram;
	CFloat32VolumeData2D* m_pReconstruction;

	CFloat32ProjectionData2D* m_pTotalRayLength;
	CFloat32VolumeData2D* m_pTotalPixelWeight;

	float m_fRelaxation;

public:

	FORCEINLINE SIRTBPPolicy();
	FORCEINLINE SIRTBPPolicy(CFloat32VolumeData2D* _pReconstruction, CFloat32ProjectionData2D* _pSinogram, CFloat32VolumeData2D* _pTotalPixelWeight, CFloat32ProjectionData2D* _pTotalRayLength, float _fRelaxation);
	FORCEINLINE ~SIRTBPPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};


//----------------------------------------------------------------------------------------
/** Policy For Sinogram Mask
 */
class SinogramMaskPolicy {

	CFloat32ProjectionData2D* m_pSinogramMask;

public:

	FORCEINLINE SinogramMaskPolicy();
	FORCEINLINE SinogramMaskPolicy(CFloat32ProjectionData2D* _pSinogramMask);
	FORCEINLINE ~SinogramMaskPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------
/** Policy For Reconstruction Mask
 */
class ReconstructionMaskPolicy {

	CFloat32VolumeData2D* m_pReconstructionMask;

public:

	FORCEINLINE ReconstructionMaskPolicy();
	FORCEINLINE ReconstructionMaskPolicy(CFloat32VolumeData2D* _pReconstructionMask);
	FORCEINLINE ~ReconstructionMaskPolicy();

	FORCEINLINE bool rayPrior(int _iRayIndex);
	FORCEINLINE bool pixelPrior(int _iVolumeIndex);
	FORCEINLINE void addWeight(int _iRayIndex, int _iVolumeIndex, float32 weight);
	FORCEINLINE void rayPosterior(int _iRayIndex);
	FORCEINLINE void pixelPosterior(int _iVolumeIndex);
};

//----------------------------------------------------------------------------------------

#include "DataProjectorPolicies.inl"

} // end namespace

#endif
