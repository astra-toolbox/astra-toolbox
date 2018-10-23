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

#ifndef _INC_ASTRA_FLOAT32DATA3DGPU
#define _INC_ASTRA_FLOAT32DATA3DGPU

#ifdef ASTRA_CUDA

#include "Globals.h"
#include "Float32Data3D.h"

#include "cuda/3d/mem3d.h"

namespace astra {


/** 
 * This class represents a 3-dimensional block of 32-bit floating point data.
 * The data block is stored on a GPU, and owned by external code.
 *
 * TODO: Store/remember which GPU the data is stored on
 */
class _AstraExport CFloat32Data3DGPU : public virtual CFloat32Data3D {

protected:
	/** Handle for the memory block */
	astraCUDA3d::MemHandle3D m_hnd;

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 */
	void _clear();

	/** Un-initialize the object, bringing it back in the unitialized state.
	 */
	void _unInit();

	/** Initialization. Initializes an instance of the CFloat32Data3DGPU class.
	 * Can only be called by derived classes.
	 *
	 * This function does not set m_bInitialized to true if everything is ok.
	 *
	 * @param _iWidth width of the 3D data (x-axis), must be > 0
	 * @param _iHeight height of the 3D data (y-axis), must be > 0
	 * @param _iDepth depth of the 3D data (z-axis), must be > 0
	 * @param _hnd the CUDA memory handle
	 */

	bool _initialize(int _iWidth, int _iHeight, int _iDepth, astraCUDA3d::MemHandle3D _hnd);

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the initialize() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 *
	 */
	CFloat32Data3DGPU();

	/** Destructor.
	 */
	virtual ~CFloat32Data3DGPU();

	/** which type is this class?
	 *
	 * @return DataType: ASTRA_DATATYPE_FLOAT32_PROJECTION or
	 *					 ASTRA_DATATYPE_FLOAT32_VOLUME
	 */
	virtual EDataType getType() const { return BASE; }

	astraCUDA3d::MemHandle3D getHandle() const { return m_hnd; }

};

} // end namespace astra

#endif

#endif // _INC_ASTRA_FLOAT32DATA3DGPU
