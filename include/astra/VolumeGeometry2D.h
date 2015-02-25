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

#ifndef _INC_ASTRA_VOLUMEGEOMETRY2D
#define _INC_ASTRA_VOLUMEGEOMETRY2D

#include "Globals.h"
#include "Config.h"

namespace astra
{

/** 
 * This class represents a pixel grid that is placed in the geometry. It defines a rectangular volume window.
 *
 * \par XML Configuration
 * \astra_xml_item{GridColCount, integer, Number of columns in this geometry.}
 * \astra_xml_item{GridRowCount, integer, Number of rows in this geometry.}
 * \astra_xml_item_option{WindowMinX, float, Minimal X-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMaxX, float, Maximal X-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMinY, float, Minimal Y-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMaxY, float, Maximal Y-coordinate in the volume window.}
 *
 * \par MATLAB example
 * \astra_code{
 *		vol_geom = struct();\n
 *		vol_geom.GridColCount = 1024;\n
 *		vol_geom.GridRowCount = 768;\n
 *		vol_geom.option.WindowMinX = -512;\n
 *		vol_geom.option.WindowMaxX = -384;\n
 *		vol_geom.option.WindowMinY = 512;\n 
 *		vol_geom.option.WindowMaxY = 384;\n
 * }
 */
class _AstraExport CVolumeGeometry2D {

protected:
	bool m_bInitialized;        ///< Has this object been initialized?

	int m_iGridColCount;		///< number of columns in the volume grid.
	int m_iGridRowCount;		///< number of rows in the volume grid.
	int m_iGridTotCount;		///< total number of pixels in the volume grid (= m_iGridColCount * m_iGridRowCount).

	/** Width of the volume window, in unit lengths.
	 *
	 * Note that this width is independent of the number of pixels in the X-direction, as the width of a pixel can 
	 * be different from 1.
	 */
	float32 m_fWindowLengthX;      

	/** Height of the volume window, in unit lengths.
	 *
	 * Note that this height is independent of the number of pixels in the Y-direction, as the height of a pixel can 
	 * be different from 1.
	 */
	float32 m_fWindowLengthY; 

	float32 m_fWindowArea;      ///< Total area of the volume window, in unit lengths squared.

	float32 m_fPixelLengthX;	///< Width of a single pixel, in unit lengths.
	float32 m_fPixelLengthY;    ///< Height of a single pixel, in unit lengths.
	float32 m_fPixelArea;       ///< Area of a single pixel, in unit lengths squared.

	float32 m_fDivPixelLengthX; ///< 1/m_fPixelLengthX, used for fast division.
	float32 m_fDivPixelLengthY; ///< 1/m_fPixelLengthY, used for fast division.

	float32 m_fWindowMinX;		///< Minimal X-coordinate in the volume window.
	float32 m_fWindowMinY;      ///< Maximal X-coordinate in the volume window.
	float32 m_fWindowMaxX;      ///< Minimal Y-coordinate in the volume window.
	float32 m_fWindowMaxY;      ///< Maximal Y-coordinate in the volume window. 

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - number of rows and columns is larger than zero
	 * - window minima is smaller than window maxima
	 * - m_iGridTotCount, m_fWindowLengthX, m_fWindowLengthY, m_fWindowArea, m_fPixelLengthX, 
	 *   m_fPixelLengthY, m_fPixelArea, m_fDivPixelLengthX and m_fDivPixelLengthY are initialized ok
	 */
	bool _check();


	/** Calculate values of all member variables from m_iGridRow/ColCount, m_fWindow*
	 */
	void _calculateDependents();
public:

    /** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 */
	CVolumeGeometry2D();

	/** Constructor. Create an instance of the CVolumeGeometry2D class. 
	 * The minimal and coordinates values of the geometry will be set to -/+ the number of rows/columns.
	 *
	 * @param _iGridCountX Number of columns in the volume grid.
	 * @param _iGridCountY Number of rows in the volume grid.
	 */
	CVolumeGeometry2D(int _iGridCountX, 
					  int _iGridCountY);

	/** Constructor. Create an instance of the CVolumeGeometry2D class.
	 *
	 * @param _iGridCountX Number of columns in the volume grid.
	 * @param _iGridCountY Number of rows in the volume grid.
	 * @param _fWindowMinX Minimal X-coordinate in the volume window.
	 * @param _fWindowMinY Minimal Y-coordinate in the volume window.
	 * @param _fWindowMaxX Maximal X-coordinate in the volume window.
	 * @param _fWindowMaxY Maximal Y-coordinate in the volume window.
	 */
	CVolumeGeometry2D(int _iGridCountX, 
					  int _iGridCountY, 
					  float32 _fWindowMinX, 
					  float32 _fWindowMinY, 
					  float32 _fWindowMaxX, 
					  float32 _fWindowMaxY);

	/** Destructor.
	 */
	virtual ~CVolumeGeometry2D();

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	*/
	void clear();

	/** Create a hard copy. 
	*/
	CVolumeGeometry2D* clone();

	/** Initialize the volume geometry with a config object.
	 *
	 * @param _cfg Configuration Object.
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialization. Initializes an instance of the CVolumeGeometry2D class.
	 * The minimal and maximal coordinates of the geometry will be set to -/+ half the number of rows/columns.
	 *
	 * If the object has been initialized before, the object is reinitialized and 
	 * memory is freed and reallocated if necessary.
	 *
	 * @param _iGridColCount Number of columns in the volume grid.
	 * @param _iGridRowCount Number of rows in the volume grid.
	 * @return initialization successful
	 */
	bool initialize(int _iGridColCount, int _iGridRowCount);

	/** Initialization. Initializes an instance of the CVolumeGeometry2D class.
	 *
	 * If the object has been initialized before, the object is reinitialized and 
	 * memory is freed and reallocated if necessary.
	 *
	 * @param _iGridColCount Number of columns in the volume grid.
	 * @param _iGridRowCount Number of rows in the volume grid.
	 * @param _fWindowMinX Minimal X-coordinate in the volume window.
	 * @param _fWindowMinY Minimal Y-coordinate in the volume window.
	 * @param _fWindowMaxX Maximal X-coordinate in the volume window.
	 * @param _fWindowMaxY Maximal Y-coordinate in the volume window.
	 * @return initialization successful
	 */
	bool initialize(int _iGridColCount, 
					int _iGridRowCount, 
					float32 _fWindowMinX, 
					float32 _fWindowMinY, 
					float32 _fWindowMaxX, 
					float32 _fWindowMaxY);

	/** Get the initialization state of the object.
	 *
	 * @return true iff the object has been initialized.
	 */
	bool isInitialized() const;

    /** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(CVolumeGeometry2D*) const;

	/** Get all settings in a Config object.
	 *
	 * @return Configuration Object.
	 */
	virtual Config* getConfiguration() const;

	/** Get the number of columns in the volume grid.
	 *
	 * @return Number of columns in the volume grid.
	 */
	int getGridColCount() const;
	
	/** Get the number of rows in the volume grid.
	 *
	 * @return Number of rows in the volume grid.
	 */
	int getGridRowCount() const;

	/** Get the total number of pixels in the volume grid.
	 *
	 * @return Total number of pixels.
	 */
	int getGridTotCount() const;

	/** Get the horizontal length of the volume window, in unit lengths.
	 *
	 * @return Horizontal length of the volume window.
	 */
	float32 getWindowLengthX() const;

	/** Get the vertical length of the volume window, in unit lengths.
	 *
	 * @return Vertical length of the volume window.
	 */
	float32 getWindowLengthY() const;

	/** Get the total area of the volume window, in unit lengths squared.
	 *
	 * @return Total area of the volume window.
	 */
	float32 getWindowArea() const;

	/** Get the horizontal length of a single pixel (i.e., width), in unit lengths.
	 *
	 * @return Horizontal length of a single pixel.
	 */
	float32 getPixelLengthX() const;

	/** Get the vertical length of a single pixel (i.e., height), in unit lengths.
	 *
	 * @return Vertical length of a single pixel.
	 */
	float32 getPixelLengthY() const;

	/** Get the area of a single pixel (width*height), in unit lengths squared.
	 *
	 * @return Area of a single pixel.
	 */
	float32 getPixelArea() const;

	/** Get the minimal X-coordinate in the volume window.
	 *
	 * @return Minimal X-coordinate in the volume window.
	 */
	float32 getWindowMinX() const;

	/** Get the minimal Y-coordinate in the volume window.
	 *
	 * @return Minimal Y-coordinate in the volume window.
	 */
	float32 getWindowMinY() const;

	/** Get the maximal X-coordinate in the volume window.
	 *
	 * @return Maximal X-coordinate in the volume window.
	 */
	float32 getWindowMaxX() const;

	/** Get the maximal Y-coordinate in the volume window.
	 *
	 * @return Maximal Y-coordinate in the volume window.
	 */
	float32 getWindowMaxY() const;

	/** Convert column and row index of a pixel to a single index in the interval [0..getGridTotCount()-1].
	 * 
	 * @param _iPixelCol Column index of the pixel, in the interval [0..getGridColCount()-1].
	 * @param _iPixelRow Row index of the pixel, in the interval [0..getGridRowCount()-1].
	 * @return Computed index of the pixel, in the interval [0..getGridTotCount()-1].
     */
	int pixelRowColToIndex(int _iPixelRow, int _iPixelCol) const;

	/** Convert a pixel index (from the interval [0..getGridTotCount()-1] to a column and row index.
	 *
	 * @param _iPixelIndex Index of the pixel, in the interval [0..getGridTotCount()-1].
	 * @param _iPixelRow Computed row index of the pixel, in the interval [0..getGridRowCount()-1].
	 * @param _iPixelCol Computed column index of the pixel, in the interval [0..getGridColCount()-1].
	 */
	void pixelIndexToRowCol(int _iPixelIndex, int &_iPixelRow, int &_iPixelCol) const;

	/** Convert a pixel column index to the X-coordinate of its center.
	 *
	 * @param _iPixelCol Column index of the pixel.
	 * @return X-coordinate of the pixel center.
	 */
	float32 pixelColToCenterX(int _iPixelCol) const;

	/** Convert a pixel column index to the minimum X-coordinate of points in that column.
	 *
	 * @param _iPixelCol Column index of the pixel.
	 * @return Minimum X-coordinate.
	 */
	float32 pixelColToMinX(int _iPixelCol) const;

	/** Convert a pixel column index to the maximum X-coordinate of points in that column.
	 *
	 * @param _iPixelCol Column index of the pixel.
	 * @return Maximum X-coordinate.
	 */
	float32 pixelColToMaxX(int _iPixelCol) const;

	/** Convert a pixel row index to the Y-coordinate of its center.
	 *
	 * @param _iPixelRow Row index of the pixel.
	 * @return Y-coordinate of the pixel center.
	 */
	float32 pixelRowToCenterY(int _iPixelRow) const;

	/** Convert a pixel row index to the minimum Y-coordinate of points in that row.
	 *
	 * @param _iPixelRow Row index of the pixel.
	 * @return Minimum Y-coordinate.
	 */
	float32 pixelRowToMinY(int _iPixelRow) const;

	/** Convert a pixel row index to the maximum Y-coordinate of points in that row.
	 *
	 * @param _iPixelRow Row index of the pixel.
	 * @return Maximum Y-coordinate.
	 */
	float32 pixelRowToMaxY(int _iPixelRow) const;

	/** Convert an X-coordinate to a column index in the volume grid.
	 *
	 * @param _fCoordX X-coordinate.
	 * @return If the X-coordinate falls within a column of the volume grid, the column index is returned. 
	 * Otherwise, a value of -1 is returned.
     */
	int coordXToCol(float32 _fCoordX) const;

	/** Convert a Y-coordinate to a row index in the volume grid.
	 *
	 * @param _fCoordY Y-coordinate 
	 * @return If the Y-coordinate falls within a row of the volume grid, the row index is returned. 
	 * Otherwise, a value of -1 is returned.
     */
	int coordYToRow(float32 _fCoordY) const;

	/** Convert an X-coordinate to an offset in the volume grid.
	 *  WindowMinX is converted to 0.
	 *
	 * @param _fCoordX X-coordinate.
	 * @return The corresponding offset in the volume grid
     */
	float coordXToColF(float32 _fCoordX) const;

	/** Convert a Y-coordinate to an offset in the volume grid.
	 *  WindowMaxY is converted to 0.
	 *
	 * @param _fCoordY Y-coordinate 
	 * @return The corresponding offset in the volume grid
     */
	float coordYToRowF(float32 _fCoordY) const;


	//< For Config unused argument checking
	ConfigCheckData* configCheckData;
	friend class ConfigStackCheck<CVolumeGeometry2D>;
};



// Get the initialization state of the object.
inline bool CVolumeGeometry2D::isInitialized() const
{
	return m_bInitialized;
}

// Get the number of columns in the volume grid.
inline int CVolumeGeometry2D::getGridColCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridColCount;
}
	
// Get the number of rows in the volume grid.
inline int CVolumeGeometry2D::getGridRowCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridRowCount;
}

// Get the total number of pixels in the volume window.
inline int CVolumeGeometry2D::getGridTotCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridTotCount;
}

// Get the horizontal length of the volume window, in unit lengths.
inline float32 CVolumeGeometry2D::getWindowLengthX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowLengthX;
}

// Get the vertical length of the volume window, in unit lengths.
inline float32 CVolumeGeometry2D::getWindowLengthY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowLengthY;
}

// Get the total area of the volume window, in unit lengths squared.
inline float32 CVolumeGeometry2D::getWindowArea() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowArea;
}

// Get the horizontal length of a single pixel (i.e., width), in unit lengths.
inline float32 CVolumeGeometry2D::getPixelLengthX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelLengthX;
}

// Get the vertical length of a single pixel (i.e., height), in unit lengths.
inline float32 CVolumeGeometry2D::getPixelLengthY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelLengthY;
}

// Get the area of a single pixel (width*height), in unit lengths squared.
inline float32 CVolumeGeometry2D::getPixelArea() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelArea;
}

 // Get the minimal X-coordinate in the volume window.
inline float32 CVolumeGeometry2D::getWindowMinX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMinX;
}

 // Get the minimal Y-coordinate in the volume window.
inline float32 CVolumeGeometry2D::getWindowMinY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMinY;
}

 // Get the maximal X-coordinate in the volume window.
inline float32 CVolumeGeometry2D::getWindowMaxX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMaxX;
}

 // Get the maximal Y-coordinate in the volume window.
inline float32 CVolumeGeometry2D::getWindowMaxY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMaxY;
}

// Convert column and row index of a pixel to a single index in the interval [0..getGridCountTot()-1].
inline int CVolumeGeometry2D::pixelRowColToIndex(int _iPixelRow, int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);
	return (_iPixelRow * m_iGridColCount + _iPixelCol);
}


// Convert a pixel index (from the interval [0..getGridCountTot()-1] to a column and row index.
inline void CVolumeGeometry2D::pixelIndexToRowCol(int _iPixelIndex, int &_iPixelRow, int &_iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelIndex >= 0);
	ASTRA_ASSERT(_iPixelIndex < m_iGridTotCount);

	_iPixelCol = (_iPixelIndex % m_iGridColCount);
	_iPixelRow = (_iPixelIndex / m_iGridColCount);
}

// Convert a pixel column index to the X-coordinate of its center
inline float32 CVolumeGeometry2D::pixelColToCenterX(int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);

	return (m_fWindowMinX + (float32(_iPixelCol) + 0.5f) * m_fPixelLengthX);
}

// Convert a pixel column index to the minimum X-coordinate of points in that column
inline float32 CVolumeGeometry2D::pixelColToMinX(int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);

	return (m_fWindowMinX + float32(_iPixelCol) * m_fPixelLengthX);
}

// Convert a pixel column index to the maximum X-coordinate of points in that column
inline float32 CVolumeGeometry2D::pixelColToMaxX(int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);

	return (m_fWindowMinX + (float32(_iPixelCol) + 1.0f) * m_fPixelLengthX);
}

// Convert a pixel row index to the Y-coordinate of its center
inline float32 CVolumeGeometry2D::pixelRowToCenterY(int _iPixelRow) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);

	return (m_fWindowMaxY - (float32(_iPixelRow) + 0.5f) * m_fPixelLengthY);
}

// Convert a pixel row index to the minimum Y-coordinate of points in that row
inline float32 CVolumeGeometry2D::pixelRowToMinY(int _iPixelRow) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);

	return (m_fWindowMaxY - (float32(_iPixelRow) + 1.0f) * m_fPixelLengthY);
}

// Convert a pixel row index to the maximum Y-coordinate of points in that row
inline float32 CVolumeGeometry2D::pixelRowToMaxY(int _iPixelRow) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);

	return (m_fWindowMaxY - (float32(_iPixelRow) * m_fPixelLengthY));
}

// Convert an X-coordinate to a column index in the volume grid
inline int CVolumeGeometry2D::coordXToCol(float32 _fCoordX) const
{
	if (_fCoordX < m_fWindowMinX) return -1;
	if (_fCoordX > m_fWindowMaxX) return -1;
	
	int iCol = int((_fCoordX - m_fWindowMinX) * m_fDivPixelLengthX);
	ASTRA_ASSERT(iCol >= 0);
	ASTRA_ASSERT(iCol < m_iGridColCount);

	return iCol;
}

// Convert a Y-coordinate to a row index in the volume grid
inline int CVolumeGeometry2D::coordYToRow(float32 _fCoordY) const
{
	if (_fCoordY < m_fWindowMinY) return -1;
	if (_fCoordY > m_fWindowMaxY) return -1;
	
	int iRow = int((m_fWindowMaxY - _fCoordY) * m_fDivPixelLengthY);
	ASTRA_ASSERT(iRow >= 0);
	ASTRA_ASSERT(iRow < m_iGridRowCount);

	return iRow;
}

// Convert an X-coordinate to an offset in the volume grid
// (WindowMinX is converted to 0)
inline float CVolumeGeometry2D::coordXToColF(float32 _fCoordX) const
{
	return (_fCoordX - m_fWindowMinX) * m_fDivPixelLengthX;
}

// Convert a Y-coordinate to an offset in the volume grid
// (WindowMaxY is converted to 0)
inline float CVolumeGeometry2D::coordYToRowF(float32 _fCoordY) const
{
	return (m_fWindowMaxY - _fCoordY) * m_fDivPixelLengthY;
}



} // end namespace astra

#endif /* _INC_ASTRA_VOLUMEGEOMETRY2D */
