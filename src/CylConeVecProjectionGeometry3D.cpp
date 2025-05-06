/*
-----------------------------------------------------------------------
Copyright: 2021, CWI, Amsterdam
           2021, University of Cambridge

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------------------------------------------------------
*/

#include "astra/CylConeVecProjectionGeometry3D.h"

#include "astra/XMLConfig.h"
#include "astra/Utilities.h"
#include "astra/Logging.h"

#include <cstring>
#include <limits>

using namespace std;

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CCylConeVecProjectionGeometry3D::CCylConeVecProjectionGeometry3D() :
	CProjectionGeometry3D() 
{

}

//----------------------------------------------------------------------------------------
// Constructor.
CCylConeVecProjectionGeometry3D::CCylConeVecProjectionGeometry3D(int _iProjectionAngleCount, 
                                                                   int _iDetectorRowCount, 
                                                                   int _iDetectorColCount, 
                                                                   std::vector<SCylConeProjection> &&_pProjectionAngles)
: CProjectionGeometry3D() 
{
	initialize(_iProjectionAngleCount, 
	           _iDetectorRowCount, 
	           _iDetectorColCount, 
	           std::move(_pProjectionAngles));
}

//----------------------------------------------------------------------------------------
// Destructor.
CCylConeVecProjectionGeometry3D::~CCylConeVecProjectionGeometry3D()
{

}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCylConeVecProjectionGeometry3D::initialize(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry3D> CR("CylConeVecProjectionGeometry3D", this, _cfg);	

	// initialization of parent class
	if (!CProjectionGeometry3D::initialize(_cfg))
		return false;

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

bool CCylConeVecProjectionGeometry3D::initializeAngles(const Config& _cfg)
{
	ConfigReader<CProjectionGeometry3D> CR("CylConeVecProjectionGeometry3D", this, _cfg);	

	// Required: Vectors
	vector<double> data;
	if (!CR.getRequiredNumericalArray("Vectors", data))
		return false;
	ASTRA_CONFIG_CHECK(data.size() % 13 == 0, "CylConeVecProjectionGeometry3D", "Vectors doesn't consist of 13-tuples.");
	m_iProjectionAngleCount = data.size() / 13;
	m_ProjectionAngles.resize(m_iProjectionAngleCount);

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		SCylConeProjection& p = m_ProjectionAngles[i];
		p.fSrcX  = data[13*i +  0];
		p.fSrcY  = data[13*i +  1];
		p.fSrcZ  = data[13*i +  2];
		p.fDetUX = data[13*i +  6];
		p.fDetUY = data[13*i +  7];
		p.fDetUZ = data[13*i +  8];
		p.fDetVX = data[13*i +  9];
		p.fDetVY = data[13*i + 10];
		p.fDetVZ = data[13*i + 11];

		// NB: Unlike for (flat) cone geometry, the detector location here
		// is the *center* of the detector.
		p.fDetCX = data[13*i +  3];
		p.fDetCY = data[13*i +  4];
		p.fDetCZ = data[13*i +  5];

		p.fDetR  = data[13*i + 12];
	}

	return true;
}

//----------------------------------------------------------------------------------------
// Initialization.
bool CCylConeVecProjectionGeometry3D::initialize(int _iProjectionAngleCount, 
                                                  int _iDetectorRowCount, 
                                                  int _iDetectorColCount, 
                                                  std::vector<SCylConeProjection> &&_pProjectionAngles)
{
	m_iProjectionAngleCount = _iProjectionAngleCount;
	m_iDetectorRowCount = _iDetectorRowCount;
	m_iDetectorColCount = _iDetectorColCount;
	m_ProjectionAngles = std::move(_pProjectionAngles);

	// TODO: check?

	// success
	m_bInitialized = _check();
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Clone
CProjectionGeometry3D* CCylConeVecProjectionGeometry3D::clone() const
{
	CCylConeVecProjectionGeometry3D* res = new CCylConeVecProjectionGeometry3D();
	res->m_bInitialized				= m_bInitialized;
	res->m_iProjectionAngleCount	= m_iProjectionAngleCount;
	res->m_iDetectorRowCount		= m_iDetectorRowCount;
	res->m_iDetectorColCount		= m_iDetectorColCount;
	res->m_iDetectorTotCount		= m_iDetectorTotCount;
	res->m_fDetectorSpacingX		= m_fDetectorSpacingX;
	res->m_fDetectorSpacingY		= m_fDetectorSpacingY;
	res->m_ProjectionAngles		= m_ProjectionAngles;
	return res;
}

//----------------------------------------------------------------------------------------
// is equal
bool CCylConeVecProjectionGeometry3D::isEqual(const CProjectionGeometry3D * _pGeom2) const
{
	if (_pGeom2 == NULL) return false;

	// try to cast argument to CCylConeProjectionGeometry3D
	const CCylConeVecProjectionGeometry3D* pGeom2 = dynamic_cast<const CCylConeVecProjectionGeometry3D*>(_pGeom2);
	if (pGeom2 == NULL) return false;

	// both objects must be initialized
	if (!m_bInitialized || !pGeom2->m_bInitialized) return false;

	// check all values
	if (m_iProjectionAngleCount != pGeom2->m_iProjectionAngleCount) return false;
	if (m_iDetectorRowCount != pGeom2->m_iDetectorRowCount) return false;
	if (m_iDetectorColCount != pGeom2->m_iDetectorColCount) return false;
	if (m_iDetectorTotCount != pGeom2->m_iDetectorTotCount) return false;
	//if (m_fDetectorSpacingX != pGeom2->m_fDetectorSpacingX) return false;
	//if (m_fDetectorSpacingY != pGeom2->m_fDetectorSpacingY) return false;
	
	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		if (memcmp(&m_ProjectionAngles[i], &pGeom2->m_ProjectionAngles[i], sizeof(m_ProjectionAngles[i])) != 0) return false;
	}

	return true;
}

//----------------------------------------------------------------------------------------
// is of type
bool CCylConeVecProjectionGeometry3D::isOfType(const std::string& _sType) const
{
	 return (_sType == "cyl_cone_vec");
}

//----------------------------------------------------------------------------------------
// Get the configuration object
Config* CCylConeVecProjectionGeometry3D::getConfiguration() const 
{
	ConfigWriter CW("ProjectionGeometry3D", "cyl_cone_vec");

	CW.addInt("DetectorRowCount", m_iDetectorRowCount);
	CW.addInt("DetectorColCount", m_iDetectorColCount);

	std::vector<double> vectors;
	vectors.resize(13 * m_iProjectionAngleCount);

	for (int i = 0; i < m_iProjectionAngleCount; ++i) {
		const SCylConeProjection& p = m_ProjectionAngles[i];
		vectors[13*i +  0] = p.fSrcX;
		vectors[13*i +  1] = p.fSrcY;
		vectors[13*i +  2] = p.fSrcZ;
		vectors[13*i +  3] = p.fDetCX;
		vectors[13*i +  4] = p.fDetCY;
		vectors[13*i +  5] = p.fDetCZ;
		vectors[13*i +  6] = p.fDetUX;
		vectors[13*i +  7] = p.fDetUY;
		vectors[13*i +  8] = p.fDetUZ;
		vectors[13*i +  9] = p.fDetVX;
		vectors[13*i + 10] = p.fDetVY;
		vectors[13*i + 11] = p.fDetVZ;
		vectors[13*i + 12] = p.fDetR;
	}
	CW.addNumericalMatrix("Vectors", &vectors[0], m_iProjectionAngleCount, 13);

	return CW.getConfig();
}
//----------------------------------------------------------------------------------------

void CCylConeVecProjectionGeometry3D::getProjectedBBoxSingleAngle(int iAngle,
                                             double fXMin, double fXMax,
                                             double fYMin, double fYMax,
                                             double fZMin, double fZMax,
                                             double &fUMin, double &fUMax,
                                             double &fVMin, double &fVMax) const
{
/*

    Get angular extent by projecting bounding box corners (using projectPoint below)

    Get vertical extent by projecting bounding box corners on two planes: one behind (tangent) and one in front of detector

*/
	const SCylConeProjection& p = m_ProjectionAngles[iAngle];

	Vec3 cyla, cylc, cylb, cylaxis;
	getCylConeAxes(p, cyla, cylb, cylc, cylaxis);

	Vec3 cylaxis_n = cylaxis * (1.0 / cylaxis.norm());

	double R = p.fDetR;
	Vec3 u(p.fDetUX, p.fDetUY, p.fDetUZ); // u (tangential) direction
	Vec3 v(p.fDetVX, p.fDetVY, p.fDetVZ); // v (axial) direction
	Vec3 s(p.fSrcX, p.fSrcY, p.fSrcZ);    // source
	double fDetUT = u.norm() / R; // angular increment

	double vol_x[8];
	double vol_y[8];
	double vol_z[8];
	vol_x[0] = vol_x[1] = vol_x[2] = vol_x[3] = fXMin;
	vol_x[4] = vol_x[5] = vol_x[6] = vol_x[7] = fXMax;

	vol_y[0] = vol_y[1] = vol_y[4] = vol_y[5] = fYMin;
	vol_y[2] = vol_y[3] = vol_y[6] = vol_y[7] = fYMax;

	vol_z[0] = vol_z[2] = vol_z[4] = vol_z[6] = fZMin;
	vol_z[1] = vol_z[3] = vol_z[5] = vol_z[7] = fZMax;

	double vol_u[9];
	double vol_v[9];

	for (int i = 0; i < 8; ++i)
		projectPoint(vol_x[i], vol_y[i], vol_z[i], iAngle, vol_u[i], vol_v[i]);

#if 0
		for (int j = 0; j < 8; ++j) {
			fprintf(stderr, "BB corner: %f %f\n", vol_u[j], vol_v[j]);
		}
#endif

	vol_u[8] = 0.0;
	vol_v[8] = 0.0;

	Vec3 ray(0,0,0);
	double umin = vol_u[8];
	double umax = vol_u[8];
	double vmin = 1e20;
	double vmax = -1e20;

	double near_plane, far_plane;

	for (int j = 8; j >= 0; --j) {
		if (vol_u[j] < umin)
			umin = vol_u[j];
		if (vol_u[j] > umax)
			umax = vol_u[j];


		// TODO: streamline this by extracting some more common subroutines

		// The vertical component doesn't matter for this computation,
		// and it may be NaN if rays missed the detector, so just use zero.
		double vv = 0; // vol_v[j] + 0.5 - 0.5*m_iDetectorRowCount;

		double theta = (vol_u[j] + 0.5 - 0.5*m_iDetectorColCount) * fDetUT;

		// truncate to half circle. (In particular for +/- infinity if rays miss the detector)
		if (theta < -PI/2)
			theta = -PI/2;
		if (theta > PI/2)
			theta = PI/2;


		Vec3 x = cylc + cyla * cos(theta) + cylb * sin(theta) + v * vv;

#if 0
		fprintf(stderr, "x: %f %f %f\n", x.x, x.y, x.z);
#endif

		if (j == 8) {
			ray = (x - cylc);
			ray = ray * (1.0 / ray.norm());

			far_plane = (x - s).dot(ray);
			near_plane = (x - s).dot(ray);
#if 0
	        fprintf(stderr, "BB near far: %f %f\n", far_plane, near_plane);
#endif
		} else {

			double t = (x - s).dot(ray);
#if 0
	        fprintf(stderr, "BB near: %f\n", t);
#endif
			if (t < near_plane)
				near_plane = t;

		}
	}

	// now project all points onto near and far planes, and get v extent from those
	for (int j = 0; j < 8; ++j) {

		Vec3 x(vol_x[j], vol_y[j], vol_z[j]);

		// TODO: check me
		double v_near = (x - s).dot(cylaxis_n) * near_plane / (x - s).dot(ray) + 0.5*m_iDetectorRowCount - 0.5;
		double v_far = (x - s).dot(cylaxis_n) * far_plane / (x - s).dot(ray) + 0.5*m_iDetectorRowCount - 0.5;

#if 0
		fprintf(stderr, "BB corner near far: %f %f\n", v_near, v_far);
#endif

		if (v_near < vmin)
			vmin = v_near;
		if (v_far < vmin)
			vmin = v_far;
		if (v_near > vmax)
			vmax = v_near;
		if (v_far > vmax)
			vmax = v_far;

	}

	fUMin = umin;
	fUMax = umax;
	fVMin = vmin;
	fVMax = vmax;
}


void CCylConeVecProjectionGeometry3D::projectPoint(double fX, double fY, double fZ,
                                                 int iAngleIndex,
                                                 double &fU, double &fV) const
{
	const SCylConeProjection& p = m_ProjectionAngles[iAngleIndex];

	// ray from p.fSrcX,p.fSrcY,p.fSrcZ through fX,fY,fZ

/*
	From kernels:

		Vec3 cyla = -cross3(u, v) * (fRadius / (u.norm() * v.norm()));
		Vec3 cylc = d - cyla;
		Vec3 cylb = u * (fRadius / u.norm());

		...

		const float fDetX = fCylCX + fCylAX*fcosdu + fCylBX*fsindu + fDetVX*fDetV;



    Normalize cyla, cylb
    Project everything onto plane spanned by cyla, cylb (coordinates obtained by inner products with normalize cyla, cylb)
    Form quadratic equation:
      in such a way that the correct one of the two branches is trivial to determine
    Solve quadratic equation,
    Angular component:
    Take arcsin to get angle (arcsin since that has the right sign behaviour for the angle range we need, assuming detector is less than pi wide)

    Vertical component:
    Project onto cylinder axis
    Use same quadratic equation solution
*/

	Vec3 x(fX, fY, fZ);

	double R = p.fDetR;

	Vec3 u(p.fDetUX, p.fDetUY, p.fDetUZ); // u (tangential) direction
	Vec3 v(p.fDetVX, p.fDetVY, p.fDetVZ); // v (axial) direction
	Vec3 s(p.fSrcX, p.fSrcY, p.fSrcZ);    // source
	Vec3 d(p.fDetCX, p.fDetCY, p.fDetCZ); // center of detector

	double fDetUT = u.norm() / R; // angular increment

	Vec3 cyla = -cross3(u, v) * (R / (u.norm() * v.norm())); // radial direction
	Vec3 cylc = d - cyla;                                    // center of cylinder
	Vec3 cylb = u * (R / u.norm());                          // tangential direction

	// unit basis vectors for coordinate system based on cylinder
	Vec3 cyla_n = cyla * (1.0 / cyla.norm());
	Vec3 cylb_n = cylb * (1.0 / cylb.norm());
	Vec3 cylaxis_n = v * (1.0 / v.norm());

	// parametrize ray as s + t*r with t >= 0
	// the intersection of the ray with the cylinder we want is the one with the largest t

	double ra = (x - s).dot(cyla_n);
	double rb = (x - s).dot(cylb_n);
	double rc = (x - s).dot(cylaxis_n);

	double sca = (s - cylc).dot(cyla_n);
	double scb = (s - cylc).dot(cylb_n);
	double scc = (s - cylc).dot(cylaxis_n);

	// quadratic equation in t:
	//
	// (t * ra + sca)^2 + (t * rb + scb)^2 = R^2

	double A = ra*ra + rb*rb;
	double B = 2*(ra*sca + rb*scb);
	double C = sca*sca + scb*scb - R*R;

	double disc = B*B - 4*A*C;

	if (disc >= 0) {
		// if non-negative, ray hits the detector.

		// A is positive, and we need the higher of the two solutions for t, so +sqrt(disc)
		double t = (-B + sqrt(disc) ) / (2 * A);

		// We expect an angle between -pi/2 and pi/2 relative to the center of the detector,
		// so use the arcsin on the 'b' component of the intersection. This gives no sign/branch problems.
		double theta = asin( (t * rb + scb) / R);

		// TODO: double check this scale/shift of angle theta to index fU
		fU = theta / fDetUT + 0.5 * m_iDetectorColCount - 0.5;

		double vv = t * rc + scc;

		// TODO: double check this scale/shift of coordinate vv to index fV
		fV = vv / v.norm() + 0.5 * m_iDetectorRowCount - 0.5;

	} else {
		// If negative, ray doesn't hit the detector.
		// In this case, we determine if it passes the detector on the positive/negative u side.
		// return +/- infinity for the angle, and NaN for the vertical.

		double tu = det3(u, d-s, v);
		double tx = det3(x-s, d-s, v);
		if (tu*tx < 0) {
			fU = -INFINITY;
		} else {
			fU = INFINITY;
		}
		fV = NAN;
	}
}


//----------------------------------------------------------------------------------------

bool CCylConeVecProjectionGeometry3D::_check()
{
	// TODO
	return true;
}

} // end namespace astra
