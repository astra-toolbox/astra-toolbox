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

#ifndef _INC_ASTRA_PARALLELBEAMBLOBPROJECTOR
#define _INC_ASTRA_PARALLELBEAMBLOBPROJECTOR

#include "ParallelProjectionGeometry2D.h"
#include "Float32Data2D.h"
#include "Projector2D.h"

namespace astra
{

/** This class implements a two-dimensional projector based on a blob-kernel.
 * A more detailed description (in dutch) is available at 
 * http://www.astra.ua.ac.be/wiki/images/6/6e/Uitleg_blob_projector.pdf
 *
 * \par XML Configuration
 * type = "blob"
 * \astra_xml_item{ProjectionGeometry, xml node, The geometry of the projection.}
 * \astra_xml_item{VolumeGeometry, xml node, The geometry of the volume.}
 * \astra_xml_item{Kernel, xml node, Kernel details.  See below.}
 *
 * \par XML Configuration of the Kernel
 * \astra_xml_item{KernelSize, float, Radius of the kernel.}
 * \astra_xml_item{SampleRate, float, Sample rate of the kernel.}
 * \astra_xml_item{SampleCount, integer, Number of samples.}
 * \astra_xml_item{KernelValues, vector of float, Samples of the kernels starting at distance 0.}
 *
 * \par MATLAB example
 * \astra_code{
 *		cfg = astra_struct('blob');\n
 *		cfg.ProjectionGeometry = proj_geom;\n
 *		cfg.VolumeGeometry = vol_geom;\n
 *		cfg.Kernel.KernelSize = 2;\n
 *		cfg.Kernel.SampleRate = 0.01;\n 
 *		cfg.Kernel.SampleCount = length(0:0.01:2);\n
 *		cfg.Kernel.KernelValues = kaiserBessel(2\, 10.4\, 2\, 0:0.01:2);\n
 *		proj_id = astra_mex_projector('create'\, cfg);\n
 * }
 */
class _AstraExport CParallelBeamBlobKernelProjector2D : public CProjector2D {

protected:
	
	/** Initial clearing. Only to be used by constructors.
	 */
	virtual void _clear();

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - no NULL pointers
	 * - all sub-objects are initialized properly
	 * - blobvalues are ok
	 */
	virtual bool _check();

public:

	// type of the projector, needed to register with CProjectorFactory
	static std::string type;

	/** Default constructor.
	 */
	CParallelBeamBlobKernelProjector2D();

	/** Constructor.
	 * 
	 * @param _pProjectionGeometry		Information class about the geometry of the projection. Will be HARDCOPIED.
	 * @param _pReconstructionGeometry	Information class about the geometry of the reconstruction volume. Will be HARDCOPIED.
	 * @param _fBlobSize				Width of the blob.  In units of PixelSize.
	 * @param _fBlobSampleRate			Spacing between two blob samples. (= _fBlobSize/_iBlobSampleCount)
	 * @param _iBlobSampleCount			Number of samples.
	 * @param _pfBlobValues				Array of _iBlobSampleCount blob evaluations.
	 */
	CParallelBeamBlobKernelProjector2D(CParallelProjectionGeometry2D* _pProjectionGeometry, 
									   CVolumeGeometry2D* _pReconstructionGeometry,
									   float32 _fBlobSize,
									   float32 _fBlobSampleRate,
									   int _iBlobSampleCount,
									   float32* _pfBlobValues);
	
	/** Destructor, is virtual to show that we are aware subclass destructor are called.
	 */	
	~CParallelBeamBlobKernelProjector2D();

	/** Initialize the projector with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize the projector.
	 *
	 * @param _pProjectionGeometry		Information class about the geometry of the projection. Will be HARDCOPIED.
	 * @param _pReconstructionGeometry	Information class about the geometry of the reconstruction volume. Will be HARDCOPIED.
	 * @param _fBlobSize				Width of the blob.  In units of PixelSize.
	 * @param _fBlobSampleRate			Spacing between two blob samples. (= _fBlobSize/_iBlobSampleCount)
	 * @param _iBlobSampleCount			Number of samples.
	 * @param _pfBlobValues				Array of _iBlobSampleCount blob evaluations. Will be HARDCOPIED.
	 */
	bool initialize(CParallelProjectionGeometry2D* _pProjectionGeometry, 
					CVolumeGeometry2D* _pReconstructionGeometry, 
					float32 _fBlobSize,
					float32 _fBlobSampleRate,
					int _iBlobSampleCount,
					float32* _pfBlobValues);

	/** Clear this class.
	 */
	virtual void clear();

	/** Returns the number of weights required for storage of all weights of one projection.
	 *
	 * @param _iProjectionIndex Index of the projection (zero-based).
	 * @return Size of buffer (given in SPixelWeight elements) needed to store weighted pixels.
	 */
	virtual int getProjectionWeightsCount(int _iProjectionIndex);

	/** Compute the pixel weights for a single ray, from the source to a detector pixel. 
	 *
	 * @param _iProjectionIndex	Index of the projection 
	 * @param _iDetectorIndex	Index of the detector pixel
	 * @param _pWeightedPixels	Pointer to a pre-allocated array, consisting of _iMaxPixelCount elements
	 *							of type SPixelWeight. On return, this array contains a list of the index
	 *							and weight for all pixels on the ray.
	 * @param _iMaxPixelCount	Maximum number of pixels (and corresponding weights) that can be stored in _pWeightedPixels.
	 *							This number MUST be greater than the total number of pixels on the ray.
	 * @param _iStoredPixelCount On return, this variable contains the total number of pixels on the 
	 *                           ray (that have been stored in the list _pWeightedPixels). 
     */
	virtual void computeSingleRayWeights(int _iProjectionIndex, 
										 int _iDetectorIndex, 
										 SPixelWeight* _pWeightedPixels,
		                                 int _iMaxPixelCount, 
										 int& _iStoredPixelCount);

	/** Create a list of detectors that are influenced by point [_iRow, _iCol].
	 *
	 * @param _iRow row of the point
	 * @param _iCol column of the point
	 * @return list of SDetector2D structs
	 */
	virtual std::vector<SDetector2D> projectPoint(int _iRow, int _iCol);

	/** Policy-based projection of all rays.  This function will calculate each non-zero projection 
	 * weight and use this value for a task provided by the policy object.
	 *
	 * @param _policy Policy object.  Should contain prior, addWeight and posterior function.
	 */
	template <typename Policy>
	void project(Policy& _policy);

	/** Policy-based projection of all rays of a single projection.  This function will calculate 
	 * each non-zero projection weight and use this value for a task provided by the policy object.
	 *
	 * @param _iProjection Wwhich projection should be projected?
	 * @param _policy Policy object.  Should contain prior, addWeight and posterior function.
	 */
	template <typename Policy>
	void projectSingleProjection(int _iProjection, Policy& _policy);

	/** Policy-based projection of a single ray.  This function will calculate each non-zero 
	 * projection  weight and use this value for a task provided by the policy object.
	 *
	 * @param _iProjection Which projection should be projected?
	 * @param _iDetector Which detector should be projected?
	 * @param _policy Policy object.  Should contain prior, addWeight and posterior function.
	 */
	template <typename Policy>
	void projectSingleRay(int _iProjection, int _iDetector, Policy& _policy);

	/** Return the  type of this projector.
	 *
	 * @return identification type of this projector
	 */
	virtual std::string getType();

protected:
	
	/** Evaluate the blob kernel for a given distance from its center.
	 *
	 * @param _fDiff distance between hit point and blob center
	 * @return blob value
	 */
	float32 _getBlobValue(float32 _fDiff);

	float32 m_fBlobSize; //< Width of the blob
	float32 m_fBlobSampleRate; //< At which interval are the inserted blob values evaluated?
	int m_iBlobSampleCount; //< Number of evaluated blob samples
	float32* m_pfBlobValues; //< Evaluated blob values
	float32* m_pfBlobValuesNeg; //< Evaluated blob values

};

//----------------------------------------------------------------------------------------

inline std::string CParallelBeamBlobKernelProjector2D::getType() 
{ 
	return type; 
}

} // namespace astra

#endif 

