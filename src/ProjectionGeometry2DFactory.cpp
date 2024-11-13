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

#include "astra/ProjectionGeometry2DFactory.h"

#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/ParallelVecProjectionGeometry2D.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/SparseMatrixProjectionGeometry2D.h"

#include "astra/Logging.h"

namespace astra
{

_AstraExport std::unique_ptr<CProjectionGeometry2D> constructProjectionGeometry2D(const std::string &type)
{
	CProjectionGeometry2D* pProjGeometry = 0;
	if (type == "parallel") {
		pProjGeometry = new CParallelProjectionGeometry2D();
	} else if (type == "parallel_vec") {
		pProjGeometry = new CParallelVecProjectionGeometry2D();
	} else if (type == "fanflat") {
		pProjGeometry = new CFanFlatProjectionGeometry2D();
	} else if (type == "fanflat_vec") {
		pProjGeometry = new CFanFlatVecProjectionGeometry2D();
	} else if (type == "sparse_matrix") {
		pProjGeometry = new CSparseMatrixProjectionGeometry2D();
	} else {
		// invalid geometry type
	}

	return std::unique_ptr<CProjectionGeometry2D>(pProjGeometry);
}

}
