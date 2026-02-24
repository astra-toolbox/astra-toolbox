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

#ifndef _INC_ASTRA_CUDARECONSTRUCTIONALGORITHM2D
#define _INC_ASTRA_CUDARECONSTRUCTIONALGORITHM2D

#include "Globals.h"
#include "Config.h"

#include "ReconstructionAlgorithm2D.h"

#include "Projector2D.h"
#include "Data2D.h"

#include "astra/cuda/2d/dims.h"

namespace astraCUDA {
class ReconAlgo;
}

namespace astra {

/**
 * This is a base class for the different CUDA implementations of 2D reconstruction algorithms.
 * They don't use a Projector, and share GPUIndex and DetectorSuperSampling options.
 *
 */
class _AstraExport CCudaReconstructionAlgorithm2D : public CReconstructionAlgorithm2D {

public:
	/** Destructor.
	 */
	virtual ~CCudaReconstructionAlgorithm2D();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/**  
	 * Sets the index of the used GPU index: first GPU has index 0
	 *
	 * @param _iGPUIndex New GPU index.
	 */
	void setGPUIndex(int _iGPUIndex);

protected:
	CCudaReconstructionAlgorithm2D();

	/** Check this object.
	 *
	 * @return object initialized
	 */
	bool _check();

	/** Initialize class. For internal use only.
	 */
	bool initialize(CProjector2D* _pProjector,
	                CFloat32ProjectionData2D* _pSinogram,
	                CFloat32VolumeData2D* _pReconstruction);


	/** Set up geometry. For internal use only.
	 */
	bool setupGeometry();

	astraCUDA::SProjectorParams2D m_params;
	Geometry2DParameters m_geometry;

	int m_iGPUIndex;

	void initializeFromProjector();
	virtual bool requiresProjector() const { return false; }

	bool callFP(const CData2D *D_vol, CData2D *D_proj, float fScale);
	bool callBP(CData2D *D_vol, const CData2D *D_proj, float fScale);
};

} // end namespace

#endif
