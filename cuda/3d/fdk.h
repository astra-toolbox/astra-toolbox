/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

#ifndef _CUDA_FDK_H
#define _CUDA_FDK_H

#include "dims3d.h"

namespace astraCUDA3d {

bool FDK_PreWeight(cudaPitchedPtr D_projData,
                float fSrcOrigin, float fDetOrigin,
                float fSrcZ, float fDetZ,
                float fDetUSize, float fDetVSize, bool bShortScan,
                const SDimensions3D& dims, const float* angles);

bool FDK(cudaPitchedPtr D_volumeData,
         cudaPitchedPtr D_projData,
         float fSrcOrigin, float fDetOrigin,
         float fSrcZ, float fDetZ, float fDetUSize, float fDetVSize,
         const SDimensions3D& dims, const float* angles, bool bShortScan);


}

#endif
