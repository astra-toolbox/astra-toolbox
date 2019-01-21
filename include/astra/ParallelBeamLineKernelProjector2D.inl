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

#define policy_weight(p,rayindex,volindex,weight) do { if (p.pixelPrior(volindex)) { p.addWeight(rayindex, volindex, weight); p.pixelPosterior(volindex); } } while (false)

template <typename Policy>
void CParallelBeamLineKernelProjector2D::project(Policy& p)
{
    projectBlock_internal(0, m_pProjectionGeometry->getProjectionAngleCount(),
        0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectSingleProjection(int _iProjection, Policy& p)
{
    projectBlock_internal(_iProjection, _iProjection + 1,
        0, m_pProjectionGeometry->getDetectorCount(), p);
}

template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectSingleRay(int _iProjection, int _iDetector, Policy& p)
{
    projectBlock_internal(_iProjection, _iProjection + 1,
        _iDetector, _iDetector + 1, p);
}

inline int fast_floor(float32 f)
{
    const auto i = static_cast<int>(f);
    if(f < 0.0f) { return (i == f) ? i : i - 1; }
    else { return i; }
}

template<typename Policy, typename Helper>
inline void ProjectPixelChecked(Policy& p, Helper& h, ProjectionData const& d, int a)
{
    const auto bf = (d.delta * a) + d.b0;
    const auto b = fast_floor(bf + 0.5f);
    assert(b >= -1); assert(b <= d.bounds + 1);
    const auto offset = bf - b;
    const auto iVolumeIndex = h.VolumeIndex(a, b);

    if(offset < -d.S) //left
    {
        const auto preWeight = (offset * d.invTminSTimesLengthPerRank) + d.invTminSTimesLengthPerRankTimesT;
        if(b > 0) { policy_weight(p, d.iRayIndex, iVolumeIndex - h.NextIndex(), d.lengthPerRank - preWeight); }
        if(b >= 0 && b < d.bounds) { policy_weight(p, d.iRayIndex, iVolumeIndex, preWeight); }
    } else if(d.S < offset) //right
    {
        const auto postWeight = (offset * d.invTminSTimesLengthPerRank) - d.invTminSTimesLengthPerRankTimesS;
        if(b >= 0 && b < d.bounds) { policy_weight(p, d.iRayIndex, iVolumeIndex, d.lengthPerRank - postWeight); }
        if(b + 1 < d.bounds) { policy_weight(p, d.iRayIndex, iVolumeIndex + h.NextIndex(), postWeight); }
    } else if(b >= 0 && b < d.bounds) //centre
    {
        policy_weight(p, d.iRayIndex, iVolumeIndex, d.lengthPerRank);
    }
}

template<typename Policy, typename Helper>
FORCEINLINE void ProjectPixel(Policy& p, Helper& h, ProjectionData const& d, int a)
{
    const auto bf = (d.delta * a) + d.b0;
    const auto b = static_cast<int>(bf + 0.5f);
    assert(b >= -1); assert(b <= d.bounds + 1);
    const auto offset = bf - b;
    const auto iVolumeIndex = h.VolumeIndex(a, b);

    if(offset < -d.S) //left
    {
        const auto preWeight = (offset * d.invTminSTimesLengthPerRank) + d.invTminSTimesLengthPerRankTimesT;
        policy_weight(p, d.iRayIndex, iVolumeIndex - h.NextIndex(), d.lengthPerRank - preWeight);
        policy_weight(p, d.iRayIndex, iVolumeIndex, preWeight);
    } else if(d.S < offset) //right
    {
        const auto postWeight = (offset * d.invTminSTimesLengthPerRank) - d.invTminSTimesLengthPerRankTimesS;
        policy_weight(p, d.iRayIndex, iVolumeIndex, d.lengthPerRank - postWeight);
        policy_weight(p, d.iRayIndex, iVolumeIndex + h.NextIndex(), postWeight);
    } else
    {
        policy_weight(p, d.iRayIndex, iVolumeIndex, d.lengthPerRank);
    }
}

template<typename Policy, typename Helper>
void ProjectRange(Policy&p, Helper const& helper, int start, int end, GlobalParameters const& gp, AngleParameters const& ap)
{
    auto isin = false;
    for(int iDetector = start; iDetector < end; ++iDetector)
    {
        const auto iRayIndex = ap.iAngle * gp.detCount + iDetector;
        if(!p.rayPrior(iRayIndex)) continue;

        const auto Dx = ap.proj->fDetSX + (iDetector + 0.5f) * ap.proj->fDetUX;
        const auto Dy = ap.proj->fDetSY + (iDetector + 0.5f) * ap.proj->fDetUY;
        const auto b0 = helper.GetB0(gp, ap, Dx, Dy);
        const auto bounds = helper.GetBounds(gp, ap, b0);
        if(bounds.StartStep == bounds.EndPost) { if(isin) { break; } else { continue; } }
        isin = true;
        const auto data = helper.GetProjectionData(gp, ap, iRayIndex, b0);

        for(auto a = bounds.StartStep; a < bounds.EndPre; ++a)
        {
            ProjectPixelChecked(p, helper, data, a);
        }
        for(auto a = bounds.EndPre; a < bounds.EndMain; ++a)
        {
            ProjectPixel(p, helper, data, a);
        }
        for(auto a = bounds.EndMain; a < bounds.EndPost; ++a)
        {
            ProjectPixelChecked(p, helper, data, a);
        }

        // POLICY: RAY POSTERIOR
        p.rayPosterior(iRayIndex);
    } // end loop detector
}

template<typename Policy, bool UseVectorization = std::is_same<DefaultFPPolicy, Policy>::value || std::is_same<DefaultBPPolicy, Policy>::value>
struct ProjectChooser;

#ifdef ENABLE_SIMD
template<typename Policy>
struct ProjectChooser<Policy, true>
{
    //FP and BP can benefit from vectorization if we cache the iRayIndex value
    template<typename Helper>
    static inline void Project(Policy&p, Helper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
    {
        const auto blockEnd = Simd::Chooser::ProjectParallelBeamLine(p, helper, gp, ap);
        ProjectRange(p, helper, blockEnd, gp.detEnd, gp, ap);
    }
};

template<typename Policy>
struct ProjectChooser<Policy, false>
#else
template<typename Policy, bool UseVectorization>
struct ProjectChooser
#endif
{
    //Plain projector for non-BP and non-FP projectors
    //These projectors don't benefit from vectorization due to being memory bound
    template<typename Helper>
    static inline void Project(Policy&p, Helper const& helper, GlobalParameters const& gp, AngleParameters const& ap)
    {
        ProjectRange(p, helper, gp.detStart, gp.detEnd, gp, ap);
    }
};

template<typename Policy>
inline void ProjectAngle(Policy& p, GlobalParameters const& gp, int iAngle, const SParProjection* proj)
{
    AngleParameters ap{gp, proj, iAngle};
    if(ap.vertical) {
        const VerticalHelper helper = {gp.colCount};
        ProjectChooser<Policy>::Project(p, helper, gp, ap);
    } else {
        const HorizontalHelper helper = {gp.colCount};
        ProjectChooser<Policy>::Project(p, helper, gp, ap);
    }
}

//----------------------------------------------------------------------------------------
/* PROJECT BLOCK - vector projection geometry

   Kernel limitations: isotropic pixels (PixelLengthX == PixelLengthY)

   For each angle/detector pair:

   Let D=(Dx,Dy) denote the centre of the detector (point) in volume coordinates, and
   let R=(Rx,Ry) denote the direction of the ray (vector).

   For mainly vertical rays (|Rx|<=|Ry|),
   let E=(Ex,Ey) denote the centre of the most upper left pixel:
      E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
   and let F=(Fx,Fy) denote a vector to the next pixel
      F = (PixelLengthX, 0)

   The intersection of the ray (D+aR) with the centre line of the upper row of pixels (E+bF) is
      { Dx + a*Rx = Ex + b*Fx
      { Dy + a*Ry = Ey + b*Fy
   Solving for (a,b) results in:
      a = (Ey + b*Fy - Dy)/Ry
        = (Ey - Dy)/Ry
      b = (Dx + a*Rx - Ex)/Fx
        = (Dx + (Ey - Dy)*Rx/Ry - Ex)/Fx

   Define c as the x-value of the intersection of the ray with the upper row in pixel coordinates.
      c = b

   The intersection of the ray (D+aR) with the centre line of the second row of pixels (E'+bF) with
      E'=(WindowMinX + PixelLengthX/2, WindowMaxY - 3*PixelLengthY/2)
   expressed in x-value pixel coordinates is
      c' = (Dx + (Ey' - Dy)*Rx/Ry - Ex)/Fx.
   And thus:
      deltac = c' - c = (Dx + (Ey' - Dy)*Rx/Ry - Ex)/Fx - (Dx + (Ey - Dy)*Rx/Ry - Ex)/Fx
                      = [(Ey' - Dy)*Rx/Ry - (Ey - Dy)*Rx/Ry]/Fx
                      = [Ey' - Ey]*(Rx/Ry)/Fx
                      = [Ey' - Ey]*(Rx/Ry)/Fx
                      = -PixelLengthY*(Rx/Ry)/Fx.

   Given c on a certain row, its closest pixel (col), and the distance (offset) to it, can be found:
      col = floor(c+1/2)
      offset = c - col

   The index of this pixel is
      volumeIndex = row * colCount + col

   The projection kernel is defined by

           _____       LengthPerRow
         /|  |  |\
        / |  |  | \
     __/  |  |  |  \__ 0
      -T -S  0  S  T

   with S = 1/2 - 1/2*|Rx/Ry|, T = 1/2 + 1/2*|Rx/Ry|, and LengthPerRow = pixelLengthX * sqrt(Rx^2+Ry^2) / |Ry|

   And thus
                              { (offset+T)/(T-S) * LengthPerRow    if  -T <= offset < S
      W_(rayIndex,volIndex) = { LengthPerRow                       if  -S <= offset <= S
                              { (offset-S)/(T-S) * LengthPerRow    if   S < offset <= T

   If -T <= offset < S, the weight for the pixel directly to the left is
      W_(rayIndex,volIndex-1) = LengthPerRow - (offset+T)/(T-S) * LengthPerRow,
   and if S < offset <= T, the weight for the pixel directly to the right is
      W_(rayIndex,volIndex+1) = LengthPerRow - (offset-S)/(T-S) * LengthPerRow.


   Mainly horizontal rays (|Rx|<=|Ry|) are handled in a similar fashion:

      E = (WindowMinX +  PixelLengthX/2, WindowMaxY - PixelLengthY/2),
      F = (0, -PixelLengthX)

      a = (Ex + b*Fx - Dx)/Rx = (Ex - Dx)/Rx
      b = (Dy + a*Ry - Ey)/Fy = (Dy + (Ex - Dx)*Ry/Rx - Ey)/Fy
      r = b
      deltar = PixelLengthX*(Ry/Rx)/Fy.
      row = floor(r+1/2)
      offset = r - row
      S = 1/2 - 1/2*|Ry/Rx|
      T = 1/2 + 1/2*|Ry/Rx|
      LengthPerCol = pixelLengthY * sqrt(Rx^2+Ry^2) / |Rx|

                              { (offset+T)/(T-S) * LengthPerCol    if  -T <= offset < S
      W_(rayIndex,volIndex) = { LengthPerCol                       if  -S <= offset <= S
                              { (offset-S)/(T-S) * LengthPerCol    if   S < offset <= T

      W_(rayIndex,volIndex-colcount) = LengthPerCol - (offset+T)/(T-S) * LengthPerCol
      W_(rayIndex,volIndex+colcount) = LengthPerCol - (offset-S)/(T-S) * LengthPerCol
*/
template <typename Policy>
void CParallelBeamLineKernelProjector2D::projectBlock_internal(int _iProjFrom, int _iProjTo, int _iDetFrom, int _iDetTo, Policy& p)
{
    // get vector geometry
    const CParallelVecProjectionGeometry2D* pVecProjectionGeometry;
    if(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)) {
        pVecProjectionGeometry = dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry)->toVectorGeometry();
    } else {
        pVecProjectionGeometry = dynamic_cast<CParallelVecProjectionGeometry2D*>(m_pProjectionGeometry);
    }

    // precomputations
    GlobalParameters gp{m_pVolumeGeometry, _iDetFrom, _iDetTo, m_pProjectionGeometry->getDetectorCount()};

    for(int iAngle = _iProjFrom; iAngle < _iProjTo; ++iAngle) {
        const SParProjection* proj = &pVecProjectionGeometry->getProjectionVectors()[iAngle];
        ProjectAngle(p, gp, iAngle, proj);
    }

    // Delete created vec geometry if required
    if(dynamic_cast<CParallelProjectionGeometry2D*>(m_pProjectionGeometry))
        delete pVecProjectionGeometry;
}