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

#include "astra/Features.h"
#include "astra/Globals.h"

namespace astra {

_AstraExport bool hasFeature(const std::string &flag) {
	if (flag == "cuda") {
		return cudaEnabled();
	}
	if (flag == "projectors_scaled_as_line_integrals") {
		return true;
	}
	if (flag == "fan_cone_BP_density_weighting_by_default") {
		return true;
	}
	if (flag == "unpadded_GPULink") {
		return true;
	}

	return false;
}

}
