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

#include "astra/ParallelBeamLineKernelProjector2D.h"

#include <cmath>
#include <algorithm>

#include "astra/DataProjectorPolicies.h"

using namespace std;
using namespace astra;

#include "astra/ParallelBeamLineKernelProjector2D.inl"

// type of the projector, needed to register with CProjectorFactory
std::string CParallelBeamLineKernelProjector2D::type = "line";

//Given a point (0, b0) with slope delta, find (an, bn)
//b(a) = delta * an + b0
//an = (bn - b0) / delta
inline int FindIntersect(float32 bn, float32 b0, float32 delta, int maxA)
{
    const auto bdiff = bn - b0;
    //If delta is really small, there may be an overflow. To avoid this,
    //check to see if the max number of steps would get us to our desired intersect
    //If not, just return the max number of steps
    if(std::abs(delta * maxA) < std::abs(bdiff)) { return maxA; }
    const auto an = bdiff / delta + 0.5f;
    const auto i_an = static_cast<int>(an);
    return i_an;
}

//This function can likely be optimized further and a single code path generated
//However, it isn't called too frequently to warrant the work
inline KernelBounds FindBounds(float32 b0, float32 delta, int maxA, int maxB)
{
    const auto lowerBoundB = -1.5f;
    const auto lowerMainB = 2.5f;
    const auto upperMainB = maxB - 2.5f;
    const auto upperBoundB = maxB + 0.5f;
    KernelBounds bounds{0, 0, 0, 0};

    if(delta < 0.0f) {
        if(b0 <= lowerBoundB) { return bounds; }
        bounds.StartStep = (b0 > upperBoundB) ? FindIntersect(upperBoundB, b0, delta, maxA) + 1 : 0;
        const auto startPos = b0 + (delta * bounds.StartStep);
        bounds.EndPre = (startPos > upperMainB) ? FindIntersect(upperMainB, startPos, delta, maxA) : 0;
        const auto prePos = startPos + (delta * bounds.EndPre);
        bounds.EndMain = (prePos > lowerMainB) ? FindIntersect(lowerMainB, prePos, delta, maxA) : 0;
        const auto mainPos = prePos + (delta * bounds.EndMain);
        bounds.EndPost = (mainPos > lowerBoundB) ? FindIntersect(lowerBoundB, mainPos, delta, maxA) : 0;
    } else {
        if(b0 >= upperBoundB) { return bounds; }
        bounds.StartStep = (b0 < lowerBoundB) ? FindIntersect(lowerBoundB, b0, delta, maxA) + 1 : 0;
        const auto startPos = b0 + (delta * bounds.StartStep);
        bounds.EndPre = (startPos < lowerMainB) ? FindIntersect(lowerMainB, startPos, delta, maxA) : 0;
        const auto prePos = startPos + (delta * bounds.EndPre);
        bounds.EndMain = (prePos < upperMainB) ? FindIntersect(upperMainB, prePos, delta, maxA) : 0;
        const auto mainPos = prePos + (delta * bounds.EndMain);
        bounds.EndPost = (mainPos < upperBoundB) ? FindIntersect(upperBoundB, mainPos, delta, maxA) : 0;
    }

    //Convert the step counts into contiguous ranges
    bounds.EndPre = std::min(bounds.EndPre + bounds.StartStep, maxA);
    bounds.EndMain = std::min(bounds.EndMain + bounds.EndPre, maxA);
    bounds.EndPost = std::min(bounds.EndPost + bounds.EndMain, maxA);
    return bounds;
}

GlobalParameters::GlobalParameters(CVolumeGeometry2D* pVolumeGeometry, int ds, int de, int dc)
	: detStart(ds), detEnd(de), detCount(dc)
{
	pixelLengthX = pVolumeGeometry->getPixelLengthX();
	pixelLengthY = pVolumeGeometry->getPixelLengthY();
	inv_pixelLengthX = 1.0f / pixelLengthX;
	inv_pixelLengthY = 1.0f / pixelLengthY;
	colCount = pVolumeGeometry->getGridColCount();
	rowCount = pVolumeGeometry->getGridRowCount();
	Ex = pVolumeGeometry->getWindowMinX() + pixelLengthX * 0.5f;
	Ey = pVolumeGeometry->getWindowMaxY() - pixelLengthY * 0.5f;
}

AngleParameters::AngleParameters(GlobalParameters const& gp, const SParProjection* p, int angle)
	: proj(p), iAngle(angle)
{
	vertical = fabs(proj->fRayX) < fabs(proj->fRayY);
	const auto detSize = sqrt(proj->fDetUX * proj->fDetUX + proj->fDetUY * proj->fDetUY);
	const auto raySize = sqrt(proj->fRayY*proj->fRayY + proj->fRayX*proj->fRayX);

	if(vertical)
	{
		RbOverRa = proj->fRayX / proj->fRayY;
		delta = -gp.pixelLengthY * RbOverRa * gp.inv_pixelLengthX;
		lengthPerRank = detSize * gp.pixelLengthX * raySize / abs(proj->fRayY);
	} else {
		RbOverRa = proj->fRayY / proj->fRayX;
		delta = -gp.pixelLengthX * RbOverRa * gp.inv_pixelLengthY;
		lengthPerRank = detSize * gp.pixelLengthY * raySize / abs(proj->fRayX);
	}
}

ProjectionData CalculateProjectionData(const int iRayIndex, const int rankCount, float32 const& b0, float32 const& delta, float32 const& RbOverRa, float32 const& lengthPerRank)
{
    const auto S = 0.5f - 0.5f * fabs(RbOverRa);
    const auto T = 0.5f + 0.5f * fabs(RbOverRa);

    const auto invTminSTimesLengthPerRank = lengthPerRank / (T - S);
    const auto invTminSTimesLengthPerRankTimesT = invTminSTimesLengthPerRank * T;
    const auto invTminSTimesLengthPerRankTimesS = invTminSTimesLengthPerRank * S;

    return {iRayIndex, rankCount, S, lengthPerRank,
        invTminSTimesLengthPerRank, invTminSTimesLengthPerRankTimesT, invTminSTimesLengthPerRankTimesS,
        b0, delta};
}

int VerticalHelper::VolumeIndex(int a, int b) const { return a * colCount + b; }
int VerticalHelper::NextIndex() const { return 1; }
std::pair<int, int> VerticalHelper::GetPixelSizes() const { return {colCount, 1}; }
float32 VerticalHelper::GetB0(GlobalParameters const& gp, AngleParameters const& ap, float32 Dx, float32 Dy) const
{
    return (Dx + (gp.Ey - Dy) * ap.RbOverRa - gp.Ex) * gp.inv_pixelLengthX;
}
KernelBounds VerticalHelper::GetBounds(GlobalParameters const& gp, AngleParameters const& ap, float32 b0) const
{
    return FindBounds(b0, ap.delta, gp.rowCount, gp.colCount);
}
ProjectionData VerticalHelper::GetProjectionData(GlobalParameters const& gp, AngleParameters const& ap, int iRayIndex, float32 b0) const
{
    return CalculateProjectionData(iRayIndex, gp.colCount, b0, ap.delta, ap.RbOverRa, ap.lengthPerRank);
}

int HorizontalHelper::VolumeIndex(int a, int b) const { return b * colCount + a; }
int HorizontalHelper::NextIndex() const { return colCount; }
std::pair<int, int> HorizontalHelper::GetPixelSizes() const { return {1, colCount}; }
float32 HorizontalHelper::GetB0(GlobalParameters const& gp, AngleParameters const& ap, float32 Dx, float32 Dy) const
{
    return -(Dy + (gp.Ex - Dx) * ap.RbOverRa - gp.Ey) * gp.inv_pixelLengthY;
}
KernelBounds HorizontalHelper::GetBounds(GlobalParameters const& gp, AngleParameters const& ap, float32 b0) const
{
    return FindBounds(b0, ap.delta, gp.colCount, gp.rowCount);
}
ProjectionData HorizontalHelper::GetProjectionData(GlobalParameters const& gp, AngleParameters const& ap, int iRayIndex, float32 b0) const
{
    return CalculateProjectionData(iRayIndex, gp.rowCount, b0, ap.delta, ap.RbOverRa, ap.lengthPerRank);
}

//----------------------------------------------------------------------------------------
// default constructor
CParallelBeamLineKernelProjector2D::CParallelBeamLineKernelProjector2D()
{
	_clear();
}

//----------------------------------------------------------------------------------------
// constructor
CParallelBeamLineKernelProjector2D::CParallelBeamLineKernelProjector2D(CParallelProjectionGeometry2D* _pProjectionGeometry,
																	   CVolumeGeometry2D* _pReconstructionGeometry)

{
	_clear();
	initialize(_pProjectionGeometry, _pReconstructionGeometry);
}

//----------------------------------------------------------------------------------------
// destructor
CParallelBeamLineKernelProjector2D::~CParallelBeamLineKernelProjector2D()
{
	clear();
}

//---------------------------------------------------------------------------------------
// Clear - Constructors
void CParallelBeamLineKernelProjector2D::_clear()
{
	CProjector2D::_clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Clear - Public
void CParallelBeamLineKernelProjector2D::clear()
{
	CProjector2D::clear();
	m_bIsInitialized = false;
}

//---------------------------------------------------------------------------------------
// Check
bool CParallelBeamLineKernelProjector2D::_check()
{
	// check base class
	ASTRA_CONFIG_CHECK(CProjector2D::_check(), "ParallelBeamLineKernelProjector2D", "Error in Projector2D initialization");

	ASTRA_CONFIG_CHECK(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry) || dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry), "ParallelBeamLineKernelProjector2D", "Unsupported projection geometry");

	ASTRA_CONFIG_CHECK(abs(m_pVolumeGeometry->getPixelLengthX() / m_pVolumeGeometry->getPixelLengthY()) - 1 < eps, "ParallelBeamLineKernelProjector2D", "Pixel height must equal pixel width.");

	// success
	return true;
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CParallelBeamLineKernelProjector2D::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector2D::initialize(_cfg)) {
		return false;
	}

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//---------------------------------------------------------------------------------------
// Initialize
bool CParallelBeamLineKernelProjector2D::initialize(CParallelProjectionGeometry2D* _pProjectionGeometry, 
													CVolumeGeometry2D* _pVolumeGeometry)
{
	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// hardcopy geometries
	m_pProjectionGeometry = _pProjectionGeometry->clone();
	m_pVolumeGeometry = _pVolumeGeometry->clone();

	// success
	m_bIsInitialized = _check();
	return m_bIsInitialized;
}

//----------------------------------------------------------------------------------------
// Get maximum amount of weights on a single ray
int CParallelBeamLineKernelProjector2D::getProjectionWeightsCount(int _iProjectionIndex)
{
	int maxDim = max(m_pVolumeGeometry->getGridRowCount(), m_pVolumeGeometry->getGridColCount());
	return maxDim * 2 + 1;
}

//----------------------------------------------------------------------------------------
// Single Ray Weights
void CParallelBeamLineKernelProjector2D::computeSingleRayWeights(int _iProjectionIndex, 
																 int _iDetectorIndex, 
																 SPixelWeight* _pWeightedPixels,
																 int _iMaxPixelCount, 
																 int& _iStoredPixelCount)
{
	ASTRA_ASSERT(m_bIsInitialized);
	StorePixelWeightsPolicy p(_pWeightedPixels, _iMaxPixelCount);
	projectSingleRay(_iProjectionIndex, _iDetectorIndex, p);
	_iStoredPixelCount = p.getStoredPixelCount();
}

//----------------------------------------------------------------------------------------
// Project Point
std::vector<SDetector2D> CParallelBeamLineKernelProjector2D::projectPoint(int _iRow, int _iCol)
{
	std::vector<SDetector2D> res;
	return res;

	// float32 xUL = m_pVolumeGeometry->pixelColToCenterX(_iCol) - m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	// float32 yUL = m_pVolumeGeometry->pixelRowToCenterY(_iRow) - m_pVolumeGeometry->getPixelLengthY() * 0.5f;
	// float32 xUR = m_pVolumeGeometry->pixelColToCenterX(_iCol) + m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	// float32 yUR = m_pVolumeGeometry->pixelRowToCenterY(_iRow) - m_pVolumeGeometry->getPixelLengthY() * 0.5f;
	// float32 xLL = m_pVolumeGeometry->pixelColToCenterX(_iCol) - m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	// float32 yLL = m_pVolumeGeometry->pixelRowToCenterY(_iRow) + m_pVolumeGeometry->getPixelLengthY() * 0.5f;
	// float32 xLR = m_pVolumeGeometry->pixelColToCenterX(_iCol) + m_pVolumeGeometry->getPixelLengthX() * 0.5f;
	// float32 yLR = m_pVolumeGeometry->pixelRowToCenterY(_iRow) + m_pVolumeGeometry->getPixelLengthY() * 0.5f;

	// std::vector<SDetector2D> res;
	// // loop projectors and detectors
	// for (int iProjection = 0; iProjection < m_pProjectionGeometry->getProjectionAngleCount(); ++iProjection) {

	// 	// get projection angle
	// 	float32 theta = m_pProjectionGeometry->getProjectionAngle(iProjection);
	// 	if (theta >= 7*PIdiv4) theta -= 2*PI;
	// 	bool inverse = false;
	// 	if (theta >= 3*PIdiv4) {
	// 		theta -= PI;
	// 		inverse = true;
	// 	}

	// 	// calculate distance from the center of the voxel to the ray though the origin
	// 	float32 tUL = xUL * cos(theta) + yUL * sin(theta);
	// 	float32 tUR = xUR * cos(theta) + yUR * sin(theta);
	// 	float32 tLL = xLL * cos(theta) + yLL * sin(theta);
	// 	float32 tLR = xLR * cos(theta) + yLR * sin(theta);
	// 	if (inverse) {
	// 		tUL *= -1.0f;
	// 		tUR *= -1.0f;
	// 		tLL *= -1.0f;
	// 		tLR *= -1.0f;
	// 	}
	// 	float32 tMin = min(tUL, min(tUR, min(tLL,tLR)));
	// 	float32 tMax = max(tUL, max(tUR, max(tLL,tLR)));

	// 	// calculate the offset on the detectorarray (in indices)
	// 	int dmin = (int)floor(m_pProjectionGeometry->detectorOffsetToIndexFloat(tMin));
	// 	int dmax = (int)ceil(m_pProjectionGeometry->detectorOffsetToIndexFloat(tMax));

	// 	// add affected detectors to the list
	// 	for (int i = dmin; i <= dmax; ++i) {
	// 		if (i >= 0 && i < m_pProjectionGeometry->getDetectorCount()) {
	// 			SDetector2D det;
	// 			det.m_iAngleIndex = iProjection;
	// 			det.m_iDetectorIndex = i;
	// 			det.m_iIndex = iProjection * getProjectionGeometry()->getDetectorCount() + i;
	// 			res.push_back(det);
	// 		}
	// 	}
	// }

	// // return result vector
	// return res;
}

//----------------------------------------------------------------------------------------
