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

#ifndef _INC_ASTRA_PROJECTORTYPELIST
#define _INC_ASTRA_PROJECTORTYPELIST

#include "Projector2D.h"
#include "TypeList.h"

// Projector2D
#include "Projector2D.h"
#include "ParallelBeamLineKernelProjector2D.h"
#include "ParallelBeamLinearKernelProjector2D.h"
#include "ParallelBeamDistanceDrivenProjector2D.h"
#include "ParallelBeamBlobKernelProjector2D.h"
#include "ParallelBeamStripKernelProjector2D.h"
#include "SparseMatrixProjector2D.h"
#include "FanFlatBeamLineKernelProjector2D.h"
#include "FanFlatBeamStripKernelProjector2D.h"
#include "CudaProjector2D.h"

namespace astra{

#ifdef ASTRA_CUDA

	typedef TYPELIST_9(
				CFanFlatBeamLineKernelProjector2D,
				CFanFlatBeamStripKernelProjector2D,
				CParallelBeamDistanceDrivenProjector2D,
				CParallelBeamLinearKernelProjector2D,
				CParallelBeamLineKernelProjector2D,
				CParallelBeamBlobKernelProjector2D,
				CParallelBeamStripKernelProjector2D, 
				CSparseMatrixProjector2D,
				CCudaProjector2D)
		Projector2DTypeList;

#else

	typedef TYPELIST_8(
				CFanFlatBeamLineKernelProjector2D,
				CFanFlatBeamStripKernelProjector2D,
				CParallelBeamDistanceDrivenProjector2D,
				CParallelBeamLinearKernelProjector2D,
				CParallelBeamLineKernelProjector2D,
				CParallelBeamBlobKernelProjector2D,
				CParallelBeamStripKernelProjector2D, 
				CSparseMatrixProjector2D)
		Projector2DTypeList;

#endif

}

// Projector3D
#include "Projector3D.h"
#include "CudaProjector3D.h"

namespace astra {

#ifdef ASTRA_CUDA

	typedef TYPELIST_1(
				CCudaProjector3D
			)
			Projector3DTypeList;
#else

	typedef TYPELIST_0 Projector3DTypeList;

#endif

}

#endif
