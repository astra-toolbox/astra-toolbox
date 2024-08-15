
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

#ifndef _INC_ASTRA_SHEPPLOGAN
#define _INC_ASTRA_SHEPPLOGAN

#include "Globals.h"
#include "Config.h"

namespace astra {

class CFloat32VolumeData2D;
class CFloat32VolumeData3D;

_AstraExport void generateSheppLogan(CFloat32VolumeData2D *data, bool modified);
_AstraExport void generateSheppLogan3D(CFloat32VolumeData3D *data, bool modified);

}

#endif
