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

#ifndef _INC_ASTRA_CUDAEMALGORITHM
#define _INC_ASTRA_CUDAEMALGORITHM

#include "Globals.h"
#include "Config.h"

#include "CudaReconstructionAlgorithm2D.h"

#ifdef ASTRA_CUDA

namespace astra {

class _AstraExport CCudaEMAlgorithm : public CCudaReconstructionAlgorithm2D
{
	
public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code.
	 */
	CCudaEMAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CCudaEMAlgorithm();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class.
	 *
	 * @param _pProjector		Projector Object. (Optional)
	 * @param _pSinogram		ProjectionData2D object containing the sinogram data.
	 * @param _pReconstruction	VolumeData2D object for storing the reconstructed volume.
	 */
	bool initialize(CProjector2D* _pProjector,
	                CFloat32ProjectionData2D* _pSinogram,
	                CFloat32VolumeData2D* _pReconstruction);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;
};

// inline functions
inline std::string CCudaEMAlgorithm::description() const { return CCudaEMAlgorithm::type; };

} // end namespace

#endif // ASTRA_CUDA

#endif 
