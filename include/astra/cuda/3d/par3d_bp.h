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

#ifndef _CUDA_PAR3D_BP_H
#define _CUDA_PAR3D_BP_H

namespace astraCUDA3d {

_AstraExport bool Par3DBP_Array(cudaPitchedPtr D_volumeData,
                   cudaArray *D_projArray,
                   const SDimensions3D& dims, const SPar3DProjection* angles,
                   const SProjectorParams3D& params);

_AstraExport bool Par3DBP(cudaPitchedPtr D_volumeData,
             cudaPitchedPtr D_projData,
             const SDimensions3D& dims, const SPar3DProjection* angles,
             const SProjectorParams3D& params);
         

}

#endif
