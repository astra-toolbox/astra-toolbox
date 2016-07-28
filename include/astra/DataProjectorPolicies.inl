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

#ifndef _INC_ASTRA_DATAPROJECTORPOLICIES_INLINE
#define _INC_ASTRA_DATAPROJECTORPOLICIES_INLINE



//----------------------------------------------------------------------------------------
// DEFAULT FORWARD PROJECTION (Ray Driven)
//----------------------------------------------------------------------------------------
DefaultFPPolicy::DefaultFPPolicy() 
{

}
//----------------------------------------------------------------------------------------
DefaultFPPolicy::DefaultFPPolicy(CFloat32VolumeData2D* _pVolumeData, 
								 CFloat32ProjectionData2D* _pProjectionData) 
{
	m_pProjectionData = _pProjectionData;
	m_pVolumeData = _pVolumeData;
}
//----------------------------------------------------------------------------------------
DefaultFPPolicy::~DefaultFPPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool DefaultFPPolicy::rayPrior(int _iRayIndex) 
{
	m_pProjectionData->getData()[_iRayIndex] = 0.0f;
	return true;
}
//----------------------------------------------------------------------------------------
bool DefaultFPPolicy::pixelPrior(int _iVolumeIndex) 
{
	// do nothing
	return true;
}
//----------------------------------------------------------------------------------------	
void DefaultFPPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pProjectionData->getData()[_iRayIndex] += m_pVolumeData->getData()[_iVolumeIndex] * _fWeight;
}
//----------------------------------------------------------------------------------------
void DefaultFPPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void DefaultFPPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
// DEFAULT BACK PROJECTION (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
DefaultBPPolicy::DefaultBPPolicy() 
{

}
//----------------------------------------------------------------------------------------
DefaultBPPolicy::DefaultBPPolicy(CFloat32VolumeData2D* _pVolumeData, 
								 CFloat32ProjectionData2D* _pProjectionData) 
{
	m_pProjectionData = _pProjectionData;
	m_pVolumeData = _pVolumeData;
}
//----------------------------------------------------------------------------------------
DefaultBPPolicy::~DefaultBPPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool DefaultBPPolicy::rayPrior(int _iRayIndex) 
{
	// do nothing
	return true;
}
//----------------------------------------------------------------------------------------
bool DefaultBPPolicy::pixelPrior(int _iVolumeIndex) 
{
	// do nothing
	return true;
}
//----------------------------------------------------------------------------------------	
void DefaultBPPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pVolumeData->getData()[_iVolumeIndex] += m_pProjectionData->getData()[_iRayIndex] * _fWeight;
}
//----------------------------------------------------------------------------------------
void DefaultBPPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void DefaultBPPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------




//----------------------------------------------------------------------------------------
// FORWARD PROJECTION DIFFERENCE CALCULATION (Ray Driven)
//----------------------------------------------------------------------------------------
DiffFPPolicy::DiffFPPolicy() 
{

}
//----------------------------------------------------------------------------------------
DiffFPPolicy::DiffFPPolicy(CFloat32VolumeData2D* _pVolumeData, 
						   CFloat32ProjectionData2D* _pDiffProjectionData, 
						   CFloat32ProjectionData2D* _pBaseProjectionData) 
{
	m_pDiffProjectionData = _pDiffProjectionData;
	m_pBaseProjectionData = _pBaseProjectionData;
	m_pVolumeData = _pVolumeData;
}
//----------------------------------------------------------------------------------------
DiffFPPolicy::~DiffFPPolicy() 
{

}
//----------------------------------------------------------------------------------------	
 bool DiffFPPolicy::rayPrior(int _iRayIndex) 
{
	m_pDiffProjectionData->getData()[_iRayIndex] = m_pBaseProjectionData->getData()[_iRayIndex];
	return true;
}
//----------------------------------------------------------------------------------------
bool DiffFPPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void DiffFPPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pDiffProjectionData->getData()[_iRayIndex] -= m_pVolumeData->getData()[_iVolumeIndex] * _fWeight;
}
//----------------------------------------------------------------------------------------
void DiffFPPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void DiffFPPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------
// STORE PIXEL WEIGHT (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
StorePixelWeightsPolicy::StorePixelWeightsPolicy() 
{

}
//----------------------------------------------------------------------------------------
StorePixelWeightsPolicy::StorePixelWeightsPolicy(SPixelWeight* _pPixelWeights, int _iMaxPixelCount) 
{
	m_iStoredPixelCount = 0;
	m_pPixelWeights = _pPixelWeights;
	m_iMaxPixelCount = _iMaxPixelCount;
}
//----------------------------------------------------------------------------------------	
StorePixelWeightsPolicy::~StorePixelWeightsPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool StorePixelWeightsPolicy::rayPrior(int _iRayIndex) 
{
	return (m_iStoredPixelCount < m_iMaxPixelCount);
}
//----------------------------------------------------------------------------------------
bool StorePixelWeightsPolicy::pixelPrior(int _iVolumeIndex) 
{
	return (m_iStoredPixelCount < m_iMaxPixelCount);
}
//----------------------------------------------------------------------------------------	
void StorePixelWeightsPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pPixelWeights[m_iStoredPixelCount].m_fWeight = _fWeight;
	m_pPixelWeights[m_iStoredPixelCount].m_iIndex = _iVolumeIndex;
	++m_iStoredPixelCount;
}
//----------------------------------------------------------------------------------------
void StorePixelWeightsPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void StorePixelWeightsPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
int StorePixelWeightsPolicy::getStoredPixelCount()
{
	return m_iStoredPixelCount;
}
//----------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------
// TOTAL PIXEL WEIGHT MULTIPLIED BY SINOGRAM (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
TotalPixelWeightBySinogramPolicy::TotalPixelWeightBySinogramPolicy() 
{

}
//----------------------------------------------------------------------------------------
TotalPixelWeightBySinogramPolicy::TotalPixelWeightBySinogramPolicy(CFloat32ProjectionData2D* _pSinogram, 
																   CFloat32VolumeData2D* _pPixelWeight) 
{
	m_pPixelWeight = _pPixelWeight;
	m_pSinogram = _pSinogram;
}
//----------------------------------------------------------------------------------------	
TotalPixelWeightBySinogramPolicy::~TotalPixelWeightBySinogramPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool TotalPixelWeightBySinogramPolicy::rayPrior(int _iRayIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------
bool TotalPixelWeightBySinogramPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void TotalPixelWeightBySinogramPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pPixelWeight->getData()[_iVolumeIndex] += _fWeight * m_pSinogram->getData()[_iRayIndex];
}
//----------------------------------------------------------------------------------------
void TotalPixelWeightBySinogramPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void TotalPixelWeightBySinogramPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------





//----------------------------------------------------------------------------------------
// TOTAL PIXEL WEIGHT (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
TotalPixelWeightPolicy::TotalPixelWeightPolicy() 
{

}
//----------------------------------------------------------------------------------------
TotalPixelWeightPolicy::TotalPixelWeightPolicy(CFloat32VolumeData2D* _pPixelWeight) 
{
	m_pPixelWeight = _pPixelWeight;
}
//----------------------------------------------------------------------------------------	
TotalPixelWeightPolicy::~TotalPixelWeightPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool TotalPixelWeightPolicy::rayPrior(int _iRayIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------
bool TotalPixelWeightPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void TotalPixelWeightPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pPixelWeight->getData()[_iVolumeIndex] += _fWeight;
}
//----------------------------------------------------------------------------------------
void TotalPixelWeightPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void TotalPixelWeightPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------




//----------------------------------------------------------------------------------------
// TOTAL RAY LENGTH (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
TotalRayLengthPolicy::TotalRayLengthPolicy() 
{

}
//----------------------------------------------------------------------------------------
TotalRayLengthPolicy::TotalRayLengthPolicy(CFloat32ProjectionData2D* _pRayLength) 
{
	m_pRayLength = _pRayLength;
}
//----------------------------------------------------------------------------------------	
TotalRayLengthPolicy::~TotalRayLengthPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool TotalRayLengthPolicy::rayPrior(int _iRayIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------
bool TotalRayLengthPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void TotalRayLengthPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	m_pRayLength->getData()[_iRayIndex] += _fWeight; 
}
//----------------------------------------------------------------------------------------
void TotalRayLengthPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void TotalRayLengthPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------





//----------------------------------------------------------------------------------------
// COMBINE TWO POLICIES (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
template<typename P1, typename P2>
CombinePolicy<P1,P2>::CombinePolicy() 
{

}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2>
CombinePolicy<P1,P2>::CombinePolicy(P1 _policy1, P2 _policy2) 
{
	policy1 = _policy1;
	policy2 = _policy2;
}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2>
CombinePolicy<P1,P2>::~CombinePolicy() 
{

}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2>
bool CombinePolicy<P1,P2>::rayPrior(int _iRayIndex) 
{
	if (!policy1.rayPrior(_iRayIndex)) return false;
	return policy2.rayPrior(_iRayIndex);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2>
bool CombinePolicy<P1,P2>::pixelPrior(int _iVolumeIndex) 
{
	if (!policy1.pixelPrior(_iVolumeIndex)) return false;
	return policy2.pixelPrior(_iVolumeIndex);
}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2>
void CombinePolicy<P1,P2>::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	policy1.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	policy2.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2>
void CombinePolicy<P1,P2>::rayPosterior(int _iRayIndex) 
{
	policy1.rayPosterior(_iRayIndex);
	policy2.rayPosterior(_iRayIndex);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2>
void CombinePolicy<P1,P2>::pixelPosterior(int _iVolumeIndex) 
{
	policy1.pixelPosterior(_iVolumeIndex);
	policy2.pixelPosterior(_iVolumeIndex);
}
//----------------------------------------------------------------------------------------




//----------------------------------------------------------------------------------------
// COMBINE THREE POLICIES  (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3>
Combine3Policy<P1,P2,P3>::Combine3Policy() 
{

}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3>
Combine3Policy<P1,P2,P3>::Combine3Policy(P1 _policy1, P2 _policy2, P3 _policy3) 
{
	policy1 = _policy1;
	policy2 = _policy2;
	policy3 = _policy3;
}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2, typename P3>
Combine3Policy<P1,P2,P3>::~Combine3Policy() 
{

}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2, typename P3>
bool Combine3Policy<P1,P2,P3>::rayPrior(int _iRayIndex) 
{
	if (!policy1.rayPrior(_iRayIndex)) return false;
	if (!policy2.rayPrior(_iRayIndex)) return false;
	return policy3.rayPrior(_iRayIndex);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3>
bool Combine3Policy<P1,P2,P3>::pixelPrior(int _iVolumeIndex) 
{
	if (!policy1.pixelPrior(_iVolumeIndex)) return false;
	if (!policy2.pixelPrior(_iVolumeIndex)) return false;
	return policy3.pixelPrior(_iVolumeIndex);
}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2, typename P3>
void Combine3Policy<P1,P2,P3>::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	policy1.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	policy2.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	policy3.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3>
void Combine3Policy<P1,P2,P3>::rayPosterior(int _iRayIndex) 
{
	policy1.rayPosterior(_iRayIndex);
	policy2.rayPosterior(_iRayIndex);
	policy3.rayPosterior(_iRayIndex);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3>
void Combine3Policy<P1,P2,P3>::pixelPosterior(int _iVolumeIndex) 
{
	policy1.pixelPosterior(_iVolumeIndex);
	policy2.pixelPosterior(_iVolumeIndex);
	policy3.pixelPosterior(_iVolumeIndex);
}
//----------------------------------------------------------------------------------------






//----------------------------------------------------------------------------------------
// COMBINE FOUR POLICIES  (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3, typename P4>
Combine4Policy<P1,P2,P3,P4>::Combine4Policy() 
{

}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3, typename P4>
Combine4Policy<P1,P2,P3,P4>::Combine4Policy(P1 _policy1, P2 _policy2, P3 _policy3, P4 _policy4) 
{
	policy1 = _policy1;
	policy2 = _policy2;
	policy3 = _policy3;
	policy4 = _policy4;
}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2, typename P3, typename P4>
Combine4Policy<P1,P2,P3,P4>::~Combine4Policy() 
{

}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2, typename P3, typename P4>
bool Combine4Policy<P1,P2,P3,P4>::rayPrior(int _iRayIndex) 
{
	if (!policy1.rayPrior(_iRayIndex)) return false;
	if (!policy2.rayPrior(_iRayIndex)) return false;
	if (!policy3.rayPrior(_iRayIndex)) return false;
	return policy4.rayPrior(_iRayIndex);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3, typename P4>
bool Combine4Policy<P1,P2,P3,P4>::pixelPrior(int _iVolumeIndex) 
{
	if (!policy1.pixelPrior(_iVolumeIndex)) return false;
	if (!policy2.pixelPrior(_iVolumeIndex)) return false;
	if (!policy3.pixelPrior(_iVolumeIndex)) return false;
	return policy4.pixelPrior(_iVolumeIndex);
}
//----------------------------------------------------------------------------------------	
template<typename P1, typename P2, typename P3, typename P4>
void Combine4Policy<P1,P2,P3,P4>::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	policy1.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	policy2.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	policy3.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	policy4.addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3, typename P4>
void Combine4Policy<P1,P2,P3,P4>::rayPosterior(int _iRayIndex) 
{
	policy1.rayPosterior(_iRayIndex);
	policy2.rayPosterior(_iRayIndex);
	policy3.rayPosterior(_iRayIndex);
	policy4.rayPosterior(_iRayIndex);
}
//----------------------------------------------------------------------------------------
template<typename P1, typename P2, typename P3, typename P4>
void Combine4Policy<P1,P2,P3,P4>::pixelPosterior(int _iVolumeIndex) 
{
	policy1.pixelPosterior(_iVolumeIndex);
	policy2.pixelPosterior(_iVolumeIndex);
	policy3.pixelPosterior(_iVolumeIndex);
	policy4.pixelPosterior(_iVolumeIndex);
}
//----------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------
// COMBINE LIST OF EQUAL POLICIES  (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
template<typename P>
CombineListPolicy<P>::CombineListPolicy() 
{
	size = 0;
}
//----------------------------------------------------------------------------------------
template<typename P>
CombineListPolicy<P>::CombineListPolicy(std::vector<P> _policyList) 
{
	policyList = _policyList;
	size = policyList.size();
}
//----------------------------------------------------------------------------------------	
template<typename P>
CombineListPolicy<P>::~CombineListPolicy() 
{

}
//----------------------------------------------------------------------------------------
template<typename P>
void CombineListPolicy<P>::addPolicy(P _policy) 
{
	policyList.push_back(_policy);
	size = policyList.size();
}
//----------------------------------------------------------------------------------------	
template<typename P>
bool CombineListPolicy<P>::rayPrior(int _iRayIndex) 
{
	for(unsigned int i = 0; i < size; ++i) {
		if (!policyList[i].rayPrior(_iRayIndex)) return false;
	}
	return true;
}
//----------------------------------------------------------------------------------------
template<typename P>
bool CombineListPolicy<P>::pixelPrior(int _iVolumeIndex) 
{
	for(unsigned int i = 0; i < size; ++i) {
		if (!policyList[i].pixelPrior(_iVolumeIndex)) return false;
	}
	return true;
}
//----------------------------------------------------------------------------------------	
template<typename P>
void CombineListPolicy<P>::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	for(unsigned int i = 0; i < size; ++i) {
		policyList[i].addWeight(_iRayIndex, _iVolumeIndex, _fWeight);
	}
}
//----------------------------------------------------------------------------------------
template<typename P>
void CombineListPolicy<P>::rayPosterior(int _iRayIndex) 
{
	for(unsigned int i = 0; i < size; ++i) {
		policyList[i].rayPosterior(_iRayIndex);
	}
}
//----------------------------------------------------------------------------------------
template<typename P>
void CombineListPolicy<P>::pixelPosterior(int _iVolumeIndex) 
{
	for(unsigned int i = 0; i < size; ++i) {
		policyList[i].pixelPosterior(_iVolumeIndex);
	}
}
//----------------------------------------------------------------------------------------





//----------------------------------------------------------------------------------------
// EMPTY POLICY  (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
EmptyPolicy::EmptyPolicy() 
{

}
//----------------------------------------------------------------------------------------	
EmptyPolicy::~EmptyPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool EmptyPolicy::rayPrior(int _iRayIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------
bool EmptyPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void EmptyPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void EmptyPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void EmptyPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------
// SIRT BACKPROJECTION  (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
SIRTBPPolicy::SIRTBPPolicy() 
{

}
//----------------------------------------------------------------------------------------
SIRTBPPolicy::SIRTBPPolicy(CFloat32VolumeData2D* _pReconstruction, 
						   CFloat32ProjectionData2D* _pSinogram, 
						   CFloat32VolumeData2D* _pTotalPixelWeight, 
						   CFloat32ProjectionData2D* _pTotalRayLength,
						   float _fRelaxation)
{
	m_pReconstruction = _pReconstruction;
	m_pSinogram = _pSinogram;
	m_pTotalPixelWeight = _pTotalPixelWeight;
	m_pTotalRayLength = _pTotalRayLength;
	m_fRelaxation = _fRelaxation;
}
//----------------------------------------------------------------------------------------	
SIRTBPPolicy::~SIRTBPPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool SIRTBPPolicy::rayPrior(int _iRayIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------
bool SIRTBPPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void SIRTBPPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	float32 fGammaBeta = m_pTotalPixelWeight->getData()[_iVolumeIndex] * m_pTotalRayLength->getData()[_iRayIndex];
	if ((fGammaBeta > 0.001f) || (fGammaBeta < -0.001f)) {
		m_pReconstruction->getData()[_iVolumeIndex] += _fWeight * m_fRelaxation * m_pSinogram->getData()[_iRayIndex] / fGammaBeta;
	}
}
//----------------------------------------------------------------------------------------
void SIRTBPPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void SIRTBPPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------





//----------------------------------------------------------------------------------------
// SINOGRAM MASK  (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
SinogramMaskPolicy::SinogramMaskPolicy() 
{

}
//----------------------------------------------------------------------------------------
SinogramMaskPolicy::SinogramMaskPolicy(CFloat32ProjectionData2D* _pSinogramMask) 
{
	m_pSinogramMask = _pSinogramMask;
}
//----------------------------------------------------------------------------------------	
SinogramMaskPolicy::~SinogramMaskPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool SinogramMaskPolicy::rayPrior(int _iRayIndex) 
{
	return (m_pSinogramMask->getData()[_iRayIndex] != 0);
}
//----------------------------------------------------------------------------------------
bool SinogramMaskPolicy::pixelPrior(int _iVolumeIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------	
void SinogramMaskPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void SinogramMaskPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void SinogramMaskPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------





//----------------------------------------------------------------------------------------
// RECONSTRUCTION MASK (Ray+Pixel Driven)
//----------------------------------------------------------------------------------------
ReconstructionMaskPolicy::ReconstructionMaskPolicy() 
{

}
//----------------------------------------------------------------------------------------
ReconstructionMaskPolicy::ReconstructionMaskPolicy(CFloat32VolumeData2D* _pReconstructionMask) 
{
	m_pReconstructionMask = _pReconstructionMask;
}
//----------------------------------------------------------------------------------------	
ReconstructionMaskPolicy::~ReconstructionMaskPolicy() 
{

}
//----------------------------------------------------------------------------------------	
bool ReconstructionMaskPolicy::rayPrior(int _iRayIndex) 
{
	return true;
}
//----------------------------------------------------------------------------------------
bool ReconstructionMaskPolicy::pixelPrior(int _iVolumeIndex) 
{
	return (m_pReconstructionMask->getData()[_iVolumeIndex] != 0);
}
//----------------------------------------------------------------------------------------	
void ReconstructionMaskPolicy::addWeight(int _iRayIndex, int _iVolumeIndex, float32 _fWeight) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void ReconstructionMaskPolicy::rayPosterior(int _iRayIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------
void ReconstructionMaskPolicy::pixelPosterior(int _iVolumeIndex) 
{
	// nothing
}
//----------------------------------------------------------------------------------------



#endif
