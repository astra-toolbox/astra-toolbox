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

#ifndef _INC_ASTRA_VECTOR3D
#define _INC_ASTRA_VECTOR3D

#include "Globals.h"

namespace astra {

/**
 * This class defines a three-dimensional vector type.
 */
class CVector3D
{
	float32 m_fX;	///< X Coordinate
	float32 m_fY;	///< Y Coordinate
	float32 m_fZ;	///< Z Coordinate

public:
	/**
	 * Default constructor
	 */
	CVector3D();

	/**
	 * Constructor initializing member variables
	 */
	CVector3D(float32 _fX, float32 _fY, float32 _fZ);

	/**
	 * Returns the X-coordinate stored in this vector
	 */
	float32 getX() const;

	/**
	 * Returns the Y-coordinate stored in this vector
	 */
	float32 getY() const;

	/**
	 * Returns the Z-coordinate stored in this vector
	 */
	float32 getZ() const;

	/**
	 * Sets the X-coordinate stored in this vector
	 */
	void setX(float32 _fX);
	
	/**
	 * Sets the X-coordinate stored in this vector
	 */
	void setY(float32 _fY);
	
	/**
	 * Sets the X-coordinate stored in this vector
	 */
	void setZ(float32 _fZ);
};

inline CVector3D::CVector3D()
{
	m_fX = m_fY = m_fZ = 0.0f;
}

inline CVector3D::CVector3D(float32 _fX, float32 _fY, float32 _fZ)
{
	m_fX = _fX;
	m_fY = _fY;
	m_fZ = _fZ;
}

inline float32 CVector3D::getX() const
{
	return m_fX;
}

inline float32 CVector3D::getY() const
{
	return m_fY;
}

inline float32 CVector3D::getZ() const
{
	return m_fZ;
}

inline void CVector3D::setX(float32 _fX)
{
	m_fX = _fX;
}

inline void CVector3D::setY(float32 _fY)
{
	m_fY = _fY;
}

inline void CVector3D::setZ(float32 _fZ)
{
	m_fZ = _fZ;
}

}

#endif /* _INC_ASTRA_VECTOR3D */
