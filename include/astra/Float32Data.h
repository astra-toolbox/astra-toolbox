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

#ifndef _INC_ASTRA_FLOAT32DATA
#define _INC_ASTRA_FLOAT32DATA

#include "Globals.h"

namespace astra {

/**
 * This is a virtual base class for floating point data classes.
 */
class _AstraExport CFloat32Data {

protected:
	
	// Protected Member Variables
	bool m_bInitialized;	///< has the object been initialized? 
	int m_iDimensions;		///< the number of dimensions

public:

	/** 
	 * Default constructor. 
	 */
	CFloat32Data();

	/** 
	 * Destructor. Free allocated memory
	 */
	virtual ~CFloat32Data();

    /**
	 * Get the initialization state of the object.
	 *
	 * @return true iff the object has been initialized
	 */
	bool isInitialized() const;

    /**
	 * Get the number of dimensions of this object.
	 *
	 * @return number of dimensions
	 */
	virtual int getDimensionCount() const = 0;
	
};

//----------------------------------------------------------------------------------------
// Inline member functions
//----------------------------------------------------------------------------------------

// Get the initialization state of the object.
inline bool CFloat32Data::isInitialized() const
{
	return m_bInitialized;
}
//----------------------------------------------------------------------------------------

} // end namespace

#endif
