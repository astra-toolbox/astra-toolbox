/*
-----------------------------------------------------------------------
Copyright: 2010-2021, imec Vision Lab, University of Antwerp
           2014-2021, CWI, Amsterdam

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

#ifndef _CUDA_CONE_CYL_H
#define _CUDA_CONE_CYL_H

namespace astraCUDA3d {

_AstraExport bool ConeCylFP_Array(cudaArray *D_volArray,
                  cudaPitchedPtr D_projData,
                  const SDimensions3D& dims, const SCylConeProjection* angles,
                  const SProjectorParams3D& params);

_AstraExport bool ConeCylFP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SCylConeProjection* angles,
            const SProjectorParams3D& params);

_AstraExport bool ConeCylBP_Array_matched(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SCylConeProjection* angles,
                  const SProjectorParams3D& params);

_AstraExport bool ConeCylBP_matched(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SCylConeProjection* angles,
            const SProjectorParams3D& params);

_AstraExport bool ConeCylBP_Array(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SCylConeProjection* angles,
                  const SProjectorParams3D& params);

_AstraExport bool ConeCylBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SCylConeProjection* angles,
            const SProjectorParams3D& params);
         
}

#endif
