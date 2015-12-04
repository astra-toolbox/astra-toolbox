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

#ifndef _INC_ASTRA_FLOAT32DATA3D
#define _INC_ASTRA_FLOAT32DATA3D

#include "Globals.h"
#include "Float32Data.h"
#include "Float32Data2D.h"

namespace astra {

class CMPIProjector3D;	

/**
 * This class represents a three-dimensional block of float32ing point data.
 */
class _AstraExport CFloat32Data3D : public CFloat32Data {

protected:
	
	int m_iWidth;			///< width of the data (x)
	int m_iHeight;			///< height of the data (y)
	int m_iDepth;			///< depth of the data (z)
	size_t m_iSize;			///< size of the data (width*height*depth)


        CMPIProjector3D *m_pMPIProjector3D; ///< reference to the used MPIProjector, if used


	/**
	 * Compares the size of two CFloat32Data instances.
	 *
	 * @param _pA CFloat32Data3D instance A
	 * @param _pB CFloat32Data3D instance B
	 * @return True if they have the same size
	 */
	static bool _data3DSizesEqual(const CFloat32Data3D * _pA, const CFloat32Data3D * _pB);

public:

	typedef enum {BASE, PROJECTION, VOLUME} EDataType;

    /** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 */
	CFloat32Data3D();

	/** Destructor.
	 */
	virtual ~CFloat32Data3D();

	/** Get the width of the data block.
	 *
	 * @return width of the data block
	 */
	int getWidth() const;

	/** Get the height of the data block.
	 *
	 * @return height of the data block
	 */
	int getHeight() const;

	/** Get the depth of the data block.
	 *
	 * @return depth of the data block
	 */
	int getDepth() const;

	/** Get the size of the data block.
	 *
	 * @return size of the data block
	 */
	int getSize() const;

	/** Which type is this class?
	 *
	 * @return DataType: PROJECTION or VOLUME
	 */
	virtual EDataType getType() const;
	
    /** Get the number of dimensions of this object.
	 *
	 * @return number of dimensions
	 */
	int getDimensionCount() const;	

	/**
	 * Clamp data to minimum value
	 *
	 * @param _fMin minimum value
	 * @return l-value
	 */
	virtual CFloat32Data3D& clampMin(float32& _fMin) = 0;

	/**
	 * Clamp data to maximum value
	 *
	 * @param _fMax maximum value
	 * @return l-value
	 */
	virtual CFloat32Data3D& clampMax(float32& _fMax) = 0;

	/** get a description of the class
	 *
	 * @return description string
	 */
	virtual std::string description() const;

	/** Get the connected MPIProjector object.
	 *
	 * @return pointer to the MPIProjector 
	 */
	CMPIProjector3D *getMPIProjector3D() const;

	/** Set the connected MPIProjector object.
	 *
	 * @param pointer to the MPIProjector 
	 */
        void setMPIProjector3D(CMPIProjector3D* prj);

	/** Test if a MPIProjector object is associated with this object
	 *
	 * @return true if there is a MPIProjector3D set, otherwise false
	 */
	bool hasMPIProjector3D() const {return m_pMPIProjector3D != NULL;}
};
//----------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------
// Get dimension count.
inline int CFloat32Data3D::getDimensionCount() const
{
	return 3;
}

//----------------------------------------------------------------------------------------
// Get the width of the data block.
inline int CFloat32Data3D::getWidth() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iWidth;
}

//----------------------------------------------------------------------------------------
// Get the height of the data block.
inline int CFloat32Data3D::getHeight() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iHeight;
}

//----------------------------------------------------------------------------------------
// Get the height of the data block.
inline int CFloat32Data3D::getDepth() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iDepth;
}

//----------------------------------------------------------------------------------------
// Get the size of the data block.
inline int CFloat32Data3D::getSize() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iSize;
}


//----------------------------------------------------------------------------------------
// get type
inline CFloat32Data3D::EDataType CFloat32Data3D::getType() const
{
	return BASE;
}


//----------------------------------------------------------------------------------------
// get MPIProjector3D
inline CMPIProjector3D* CFloat32Data3D::getMPIProjector3D() const
{
        return m_pMPIProjector3D;
}

//----------------------------------------------------------------------------------------
// set MPIProjector3D
inline void CFloat32Data3D::setMPIProjector3D(CMPIProjector3D* prj)
{
         m_pMPIProjector3D = prj;
}



} // end namespace astra

#endif // _INC_ASTRA_FLOAT32DATA2D
