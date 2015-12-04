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

#ifndef _INC_ASTRA_VOLUMEGEOMETRY3D
#define _INC_ASTRA_VOLUMEGEOMETRY3D

#include "Globals.h"
#include "Config.h"
#include "VolumeGeometry2D.h"

namespace astra
{

/** 
 * This class represents a 3D pixel grid that is placed in the geometry. It defines a rectangular volume window.
 *
 * \par XML Configuration
 * \astra_xml_item{GridColCount, integer, Number of columns in this geometry.}
 * \astra_xml_item{GridRowCount, integer, Number of rows in this geometry.}
 * \astra_xml_item{GridSliceCount, integer, Number of slices in this geometry.}
 * \astra_xml_item_option{WindowMinX, float, -GridColCount/2, Minimal X-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMaxX, float, GridColCount/2, Maximal X-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMinY, float, -GridRowCount/2, Minimal Y-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMaxY, float, GridRowCount/2, Maximal Y-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMinZ, float, -GridSliceCount/2, Minimal Z-coordinate in the volume window.}
 * \astra_xml_item_option{WindowMaxZ, float, GridSliceCount/2, Maximal Z-coordinate in the volume window.}
 *
 * \par MATLAB example
 * \astra_code{
 *		vol_geom = struct();\n
 *		vol_geom.GridColCount = 1024;\n
 *		vol_geom.GridRowCount = 768;\n
 * 		vol_geom.GridSliceCount = 300;\n
 *		vol_geom.option.WindowMinX = -512;\n
 *		vol_geom.option.WindowMaxX = -384;\n
 *		vol_geom.option.WindowMinY = 512;\n 
 *		vol_geom.option.WindowMaxY = 384;\n
 *		vol_geom.option.WindowMinZ = -150;\n 
 *		vol_geom.option.WindowMaxZ = 150;\n
 * }
 */
class _AstraExport CVolumeGeometry3D {

protected:
	bool m_bInitialized;        ///< Has this object been initialized?

	int m_iGridColCount;		///< number of columns in the volume grid.
	int m_iGridRowCount;		///< number of rows in the volume grid.
	int m_iGridSliceCount;		///< number of slices in the volume grid.
	int m_iGridTotCount;		///< total number of pixels in the volume grid (= m_iGridColCount * m_iGridRowCount * m_iGridSliceCount).

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

	/** Depth of the volume window, in unit lengths.
	 *
	 * Note that this depth is independent of the number of pixels in the Z-direction, as the depth of a pixel can 
	 * be different from 1.
	 */
	float32 m_fWindowLengthZ; 

	/** Total area of the volume window, in unit lengths squared.
	 */
	float32 m_fWindowArea;      

	float32 m_fPixelLengthX;	///< Width of a single pixel, in unit lengths.
	float32 m_fPixelLengthY;    ///< Height of a single pixel, in unit lengths.
	float32 m_fPixelLengthZ;    ///< Depth of a single pixel, in unit lengths.
	float32 m_fPixelArea;       ///< Area of a single pixel, in unit lengths squared.

	float32 m_fDivPixelLengthX; ///< 1/m_fPixelLengthX, used for fast division.
	float32 m_fDivPixelLengthY; ///< 1/m_fPixelLengthY, used for fast division.
	float32 m_fDivPixelLengthZ; ///< 1/m_fPixelLengthZ, used for fast division.

	float32 m_fWindowMinX;		///< Minimal X-coordinate in the volume window.
	float32 m_fWindowMinY;      ///< Minimal Y-coordinate in the volume window.
	float32 m_fWindowMinZ;      ///< Minimal Z-coordinate in the volume window.
	float32 m_fWindowMaxX;      ///< Maximal X-coordinate in the volume window.
	float32 m_fWindowMaxY;      ///< Maximal Y-coordinate in the volume window. 
	float32 m_fWindowMaxZ;      ///< Maximal Z-coordinate in the volume window. 

	/** Check the values of this object.  If everything is ok, the object can be set to the initialized state.
	 * The following statements are then guaranteed to hold:
	 * - number of rows, columns and slices is larger than zero
	 * - window minima is smaller than window maxima
	 * - m_iGridTotCount, m_fWindowLengthX, m_fWindowLengthY, m_fWindowLengthZ, m_fWindowArea, m_fPixelLengthX, 
	 *   m_fPixelLengthY, m_fPixelLengthZ, m_fPixelArea, m_fDivPixelLengthX, m_fDivPixelLengthY 
	 *   and m_fDivPixelLengthZ are initialized ok
	 */
	bool _check();

public:

	/** Default constructor. Sets all numeric member variables to 0 and all pointer member variables to NULL.
	 *
	 * If an object is constructed using this default constructor, it must always be followed by a call 
	 * to one of the init() methods before the object can be used. Any use before calling init() is not allowed,
	 * except calling the member function isInitialized().
	 */
	CVolumeGeometry3D();

	/** Constructor. Create an instance of the CVolumeGeometry2D class. 
	 * The minimal and coordinates values of the geometry will be set to -/+ the number of rows/columns.
	 *
	 * @param _iGridCountX Number of columns in the volume grid.
	 * @param _iGridCountY Number of rows in the volume grid.
	 * @param _iGridCountZ Number of slices in the volume grid.
	 */
	CVolumeGeometry3D(int _iGridCountX, int _iGridCountY, int _iGridCountZ);

	/** Constructor. Create an instance of the CVolumeGeometry2D class.
	 *
	 * @param _iGridCountX Number of columns in the volume grid.
	 * @param _iGridCountY Number of rows in the volume grid.
	 * @param _iGridCountZ Number of slices in the volume grid.
	 * @param _fWindowMinX Minimal X-coordinate in the volume window.
	 * @param _fWindowMinY Minimal Y-coordinate in the volume window.
	 * @param _fWindowMinZ Minimal Z-coordinate in the volume window.
	 * @param _fWindowMaxX Maximal X-coordinate in the volume window.
	 * @param _fWindowMaxY Maximal Y-coordinate in the volume window.
	 * @param _fWindowMaxZ Maximal Z-coordinate in the volume window.
	 */
	CVolumeGeometry3D(int _iGridCountX, 
					  int _iGridCountY, 
					  int _iGridCountZ, 
					  float32 _fWindowMinX, 
					  float32 _fWindowMinY, 
					  float32 _fWindowMinZ, 
					  float32 _fWindowMaxX, 
					  float32 _fWindowMaxY,
					  float32 _fWindowMaxZ);

	/**
	 * Copy constructor
	 */
	CVolumeGeometry3D(const CVolumeGeometry3D& _other);

	/**
	 * Assignment operator
	 */
	CVolumeGeometry3D& operator=(const CVolumeGeometry3D& _other);

	/** Destructor.
	 */
	virtual ~CVolumeGeometry3D();

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	*/
	void clear();

	/** Create a hard copy. 
	*/
	CVolumeGeometry3D* clone() const;

	/** Initialize the volume geometry with a config object.
	 *
	 * @param _cfg Configuration Object.
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialization. Initializes an instance of the CVolumeGeometry3D class.
	 * The minimal and maximal coordinates of the geometry will be set to -/+ half the number of rows/columns/slices.
	 *
	 * If the object has been initialized before, the object is reinitialized and 
	 * memory is freed and reallocated if necessary.
	 *
	 * @param _iGridColCount Number of columns in the volume grid.
	 * @param _iGridRowCount Number of rows in the volume grid.
	 * @param _iGridSliceCount Number of slices in the volume grid.
	 * @return initialization successful
	 */
	bool initialize(int _iGridColCount, int _iGridRowCount, int _iGridSliceCount);

	/** Initialization. Initializes an instance of the CVolumeGeometry3D class.
	 *
	 * If the object has been initialized before, the object is reinitialized and 
	 * memory is freed and reallocated if necessary.
	 *
	 * @param _iGridColCount Number of columns in the volume grid.
	 * @param _iGridRowCount Number of rows in the volume grid.
	 * @param _iGridSliceCount Number of slices in the volume grid.
	 * @param _fWindowMinX Minimal X-coordinate in the volume window.
	 * @param _fWindowMinY Minimal Y-coordinate in the volume window.
	 * @param _fWindowMinZ Minimal Z-coordinate in the volume window.
	 * @param _fWindowMaxX Maximal X-coordinate in the volume window.
	 * @param _fWindowMaxY Maximal Y-coordinate in the volume window.
	 * @param _fWindowMaxZ Maximal Z-coordinate in the volume window.
	 * @return initialization successful
	 */
	bool initialize(int _iGridColCount, 
					int _iGridRowCount, 
					int _iGridSliceCount, 
					float32 _fWindowMinX, 
					float32 _fWindowMinY, 
					float32 _fWindowMinZ,
					float32 _fWindowMaxX, 
					float32 _fWindowMaxY,
					float32 _fWindowMaxZ);

	/** Get the initialization state of the object.
	 *
	 * @return true iff the object has been initialized.
	 */
	bool isInitialized() const;

	/** Return true if this geometry instance is the same as the one specified.
	 *
	 * @return true if this geometry instance is the same as the one specified.
	 */
	virtual bool isEqual(const CVolumeGeometry3D*) const;

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

	/** Get the number of slices in the volume grid.
	 *
	 * @return Number of slices in the volume grid.
	 */
	int getGridSliceCount() const;



	/** Set the number of slices in the volume grid.
	 *
	 */
	void setGridSliceCount(const int);


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

	/** Get the depth of the volume window, in unit lengths.
	 *
	 * @return Depth of the volume window.
	 */
	float32 getWindowLengthZ() const;

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

	/** Get the depth of a single pixel in unit lengths.
	 *
	 * @return Depth of a single pixel.
	 */
	float32 getPixelLengthZ() const;

	/** Get the area of a single pixel (width*height*depth), in unit lengths squared.
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

	/** Get the minimal Z-coordinate in the volume window.
	 *
	 * @return Minimal Z-coordinate in the volume window.
	 */
	float32 getWindowMinZ() const;

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

	/** Get the maximal Z-coordinate in the volume window.
	 *
	 * @return Maximal Z-coordinate in the volume window.
	 */
	float32 getWindowMaxZ() const;

	/** Convert row, column and slice index of a pixel to a single index in the interval [0..getGridTotCount()-1].
	 * 
	 * @param _iPixelRow Row index of the pixel, in the interval [0..getGridRowCount()-1].
	 * @param _iPixelCol Column index of the pixel, in the interval [0..getGridColCount()-1].
	 * @param _iPixelSlice Slice index of the pixel, in the interval [0..getGridSliceCount()-1].
	 * @return Computed index of the pixel, in the interval [0..getGridTotCount()-1].
	 */
	int pixelRowColSliceToIndex(int _iPixelRow, int _iPixelCol, int _iPixelSlice) const;

	/** Convert a pixel index (from the interval [0..getGridTotCount()-1] to row, column and slice index.
	 *
	 * @param _iPixelIndex Index of the pixel, in the interval [0..getGridTotCount()-1].
	 * @param _iPixelRow Computed row index of the pixel, in the interval [0..getGridRowCount()-1].
	 * @param _iPixelCol Computed column index of the pixel, in the interval [0..getGridColCount()-1].
	 * @param _iPixelSlice Computed slice index of the pixel, in the interval [0..getGridSliceCount()-1].
	 */
	void pixelIndexToRowColSlice(int _iPixelIndex, int &_iPixelRow, int &_iPixelCol, int &_iPixelSlice) const;

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

	/** Convert a pixel slice index to the Z-coordinate of its center.
	 *
	 * @param _iPixelSlice Slice index of the pixel.
	 * @return Z-coordinate of the pixel center.
	 */
	float32 pixelSliceToCenterZ(int _iPixelSlice) const;

	/** Convert a pixel slice index to the minimum Z-coordinate of points in that slice.
	 *
	 * @param _iPixelSlice Slice index of the pixel.
	 * @return Minimum Z-coordinate.
	 */
	float32 pixelSliceToMinZ(int _iPixelSlice) const;

	/** Convert a pixel slice index to the maximum Z-coordinate of points in that slice.
	 *
	 * @param _iPixelSlice Slice index of the pixel.
	 * @return Maximum Z-coordinate.
	 */
	float32 pixelSliceToMaxZ(int _iPixelSlice) const;

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

	/** Convert a Z-coordinate to a slice index in the volume grid.
	 *
	 * @param _fCoordZ Z-coordinate 
	 * @return If the Z-coordinate falls within a slice of the volume grid, the slice index is returned. 
	 * Otherwise, a value of -1 is returned.
	 */
	int coordZToSlice(float32 _fCoordZ) const;

	/** Convert an X-coordinate to a column index in the volume grid.
	 *
	 * @param _fCoordX X-coordinate.
	 * @return If the X-coordinate falls within a column of the volume grid, the column index is returned. 
	 * Otherwise, a value of -1 is returned.
	 */
	float32 coordXToColFloat(float32 _fCoordX) const;

	/** Convert a Y-coordinate to a row index in the volume grid.
	 *
	 * @param _fCoordY Y-coordinate 
	 * @return If the Y-coordinate falls within a row of the volume grid, the row index is returned. 
	 * Otherwise, a value of -1 is returned.
	 */
	float32 coordYToRowFloat(float32 _fCoordY) const;

	/** Convert a Z-coordinate to a slice index in the volume grid.
	 *
	 * @param _fCoordZ Z-coordinate 
	 * @return If the Z-coordinate falls within a slice of the volume grid, the slice index is returned. 
	 * Otherwise, a value of -1 is returned.
	 */
	float32 coordZToSliceFloat(float32 _fCoordZ) const;

	CVolumeGeometry2D * createVolumeGeometry2D() const;


	//< For Config unused argument checking
	ConfigCheckData* configCheckData;
	friend class ConfigStackCheck<CVolumeGeometry3D>;
};


//----------------------------------------------------------------------------------------
// Get the initialization state of the object.
inline bool CVolumeGeometry3D::isInitialized() const
{
	return m_bInitialized;
}

//----------------------------------------------------------------------------------------
// Get the number of columns in the volume grid.
inline int CVolumeGeometry3D::getGridColCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridColCount;
}
	
//----------------------------------------------------------------------------------------
// Get the number of rows in the volume grid.
inline int CVolumeGeometry3D::getGridRowCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridRowCount;
}

//----------------------------------------------------------------------------------------
// Get the number of rows in the volume grid.
inline int CVolumeGeometry3D::getGridSliceCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridSliceCount;
}


inline void CVolumeGeometry3D::setGridSliceCount(const int nSlices)
{
	ASTRA_ASSERT(m_bInitialized);
	m_iGridSliceCount  = nSlices;
	m_fWindowMinZ 	   = -m_iGridSliceCount/2.0f;
	m_fWindowMaxZ  	   =  m_iGridSliceCount/2.0f;
	m_iGridTotCount    = (m_iGridColCount  * m_iGridRowCount * m_iGridSliceCount);
	m_fWindowLengthZ   = (m_fWindowMaxZ    - m_fWindowMinZ);
	m_fWindowArea 	   = (m_fWindowLengthX * m_fWindowLengthY * m_fWindowLengthZ);
	m_fPixelLengthZ	   = (m_fWindowLengthZ / (float32)m_iGridSliceCount);
	m_fPixelArea 	   = (m_fPixelLengthX  * m_fPixelLengthY * m_fPixelLengthZ);
	m_fDivPixelLengthZ = ((float32)m_iGridSliceCount / m_fWindowLengthZ); // == (1.0f / m_fPixelLengthZ);
}


//----------------------------------------------------------------------------------------
// Get the total number of pixels in the volume window.
inline int CVolumeGeometry3D::getGridTotCount() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_iGridTotCount;
}

//----------------------------------------------------------------------------------------
// Get the horizontal length of the volume window, in unit lengths.
inline float32 CVolumeGeometry3D::getWindowLengthX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowLengthX;
}

//----------------------------------------------------------------------------------------
// Get the vertical length of the volume window, in unit lengths.
inline float32 CVolumeGeometry3D::getWindowLengthY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowLengthY;
}

//----------------------------------------------------------------------------------------
// Get the vertical length of the volume window, in unit lengths.
inline float32 CVolumeGeometry3D::getWindowLengthZ() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowLengthZ;
}

//----------------------------------------------------------------------------------------
// Get the total area of the volume window, in unit lengths squared.
inline float32 CVolumeGeometry3D::getWindowArea() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowArea;
}

//----------------------------------------------------------------------------------------
// Get the horizontal length of a single pixel (i.e., width), in unit lengths.
inline float32 CVolumeGeometry3D::getPixelLengthX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelLengthX;
}

//----------------------------------------------------------------------------------------
// Get the vertical length of a single pixel (i.e., height), in unit lengths.
inline float32 CVolumeGeometry3D::getPixelLengthY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelLengthY;
}

//----------------------------------------------------------------------------------------
// Get the depth of a single pixel in unit lengths.
inline float32 CVolumeGeometry3D::getPixelLengthZ() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelLengthZ;
}

//----------------------------------------------------------------------------------------
// Get the area of a single pixel (width*height), in unit lengths squared.
inline float32 CVolumeGeometry3D::getPixelArea() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fPixelArea;
}

//----------------------------------------------------------------------------------------
// Get the minimal X-coordinate in the volume window.
inline float32 CVolumeGeometry3D::getWindowMinX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMinX;
}

//----------------------------------------------------------------------------------------
// Get the minimal Y-coordinate in the volume window.
inline float32 CVolumeGeometry3D::getWindowMinY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMinY;
}

//----------------------------------------------------------------------------------------
// Get the minimal Y-coordinate in the volume window.
inline float32 CVolumeGeometry3D::getWindowMinZ() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMinZ;
}

//----------------------------------------------------------------------------------------
// Get the maximal X-coordinate in the volume window.
inline float32 CVolumeGeometry3D::getWindowMaxX() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMaxX;
}

//----------------------------------------------------------------------------------------
// Get the maximal Y-coordinate in the volume window.
inline float32 CVolumeGeometry3D::getWindowMaxY() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMaxY;
}

//----------------------------------------------------------------------------------------
// Get the maximal Z-coordinate in the volume window.
inline float32 CVolumeGeometry3D::getWindowMaxZ() const
{
	ASTRA_ASSERT(m_bInitialized);
	return m_fWindowMaxZ;
}

//----------------------------------------------------------------------------------------
// Convert row, column and slice index of a pixel to a single index in the interval [0..getGridCountTot()-1].
inline int CVolumeGeometry3D::pixelRowColSliceToIndex(int _iPixelRow, int _iPixelCol, int _iPixelSlice) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);
	ASTRA_ASSERT(_iPixelSlice >= 0);
	ASTRA_ASSERT(_iPixelSlice < m_iGridSliceCount);

	return (m_iGridColCount*m_iGridRowCount*_iPixelSlice + _iPixelRow * m_iGridColCount + _iPixelCol);
}

//----------------------------------------------------------------------------------------
// Convert a pixel index (from the interval [0..getGridCountTot()-1] to a row, column and slice index.
inline void CVolumeGeometry3D::pixelIndexToRowColSlice(int _iPixelIndex, int &_iPixelRow, int &_iPixelCol, int &_iPixelSlice) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelIndex >= 0);
	ASTRA_ASSERT(_iPixelIndex < m_iGridTotCount);

	_iPixelSlice = _iPixelIndex / (m_iGridRowCount*m_iGridColCount);
	_iPixelRow = (_iPixelIndex-_iPixelSlice*m_iGridRowCount*m_iGridColCount) / m_iGridColCount;
	_iPixelCol = (_iPixelIndex-_iPixelSlice*m_iGridRowCount*m_iGridColCount) % m_iGridColCount;
}

//----------------------------------------------------------------------------------------
// Convert a pixel column index to the X-coordinate of its center
inline float32 CVolumeGeometry3D::pixelColToCenterX(int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);

	return (m_fWindowMinX + (float32(_iPixelCol) + 0.5f) * m_fPixelLengthX);
}

//----------------------------------------------------------------------------------------
// Convert a pixel column index to the minimum X-coordinate of points in that column
inline float32 CVolumeGeometry3D::pixelColToMinX(int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);

	return (m_fWindowMinX + float32(_iPixelCol) * m_fPixelLengthX);
}

//----------------------------------------------------------------------------------------
// Convert a pixel column index to the maximum X-coordinate of points in that column
inline float32 CVolumeGeometry3D::pixelColToMaxX(int _iPixelCol) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelCol >= 0);
	ASTRA_ASSERT(_iPixelCol < m_iGridColCount);

	return (m_fWindowMaxX + (float32(_iPixelCol) + 1.0f) * m_fPixelLengthX);
}

//----------------------------------------------------------------------------------------
// Convert a pixel row index to the Y-coordinate of its center
inline float32 CVolumeGeometry3D::pixelRowToCenterY(int _iPixelRow) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);

	return (m_fWindowMaxY - (float32(_iPixelRow) + 0.5f) * m_fPixelLengthY);
}

//----------------------------------------------------------------------------------------
// Convert a pixel row index to the minimum Y-coordinate of points in that row
inline float32 CVolumeGeometry3D::pixelRowToMinY(int _iPixelRow) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);

	return (m_fWindowMaxY - (float32(_iPixelRow) + 1.0f) * m_fPixelLengthY);
}

//----------------------------------------------------------------------------------------
// Convert a pixel row index to the maximum Y-coordinate of points in that row
inline float32 CVolumeGeometry3D::pixelRowToMaxY(int _iPixelRow) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelRow >= 0);
	ASTRA_ASSERT(_iPixelRow < m_iGridRowCount);

	return (m_fWindowMaxY - (float32(_iPixelRow) * m_fPixelLengthY));
}

//----------------------------------------------------------------------------------------
// Convert a pixel slice index to the Z-coordinate of its center
inline float32 CVolumeGeometry3D::pixelSliceToCenterZ(int _iPixelSlice) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelSlice >= 0);
	ASTRA_ASSERT(_iPixelSlice < m_iGridSliceCount);

	return (m_fWindowMaxZ - (float32(_iPixelSlice) + 0.5f) * m_fPixelLengthZ);
}

//----------------------------------------------------------------------------------------
// Convert a pixel row index to the minimum Y-coordinate of points in that row
inline float32 CVolumeGeometry3D::pixelSliceToMinZ(int _iPixelSlice) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelSlice >= 0);
	ASTRA_ASSERT(_iPixelSlice < m_iGridSliceCount);

	return (m_fWindowMaxZ - (float32(_iPixelSlice) + 1.0f) * m_fPixelLengthZ);
}

//----------------------------------------------------------------------------------------
// Convert a pixel row index to the maximum Y-coordinate of points in that row
inline float32 CVolumeGeometry3D::pixelSliceToMaxZ(int _iPixelSlice) const
{
	ASTRA_ASSERT(m_bInitialized);
	ASTRA_ASSERT(_iPixelSlice >= 0);
	ASTRA_ASSERT(_iPixelSlice < m_iGridSliceCount);

	return (m_fWindowMaxZ - (float32(_iPixelSlice) * m_fPixelLengthZ));
}

//----------------------------------------------------------------------------------------
// Convert an X-coordinate to a column index in the volume grid
inline int CVolumeGeometry3D::coordXToCol(float32 _fCoordX) const
{
	if (_fCoordX < m_fWindowMinX) return -1;
	if (_fCoordX > m_fWindowMaxX) return -1;
	
	int iCol = int((_fCoordX - m_fWindowMinX) * m_fDivPixelLengthX);
	ASTRA_ASSERT(iCol >= 0);
	ASTRA_ASSERT(iCol < m_iGridColCount);

	return iCol;
}

//----------------------------------------------------------------------------------------
// Convert a Y-coordinate to a row index in the volume grid
inline int CVolumeGeometry3D::coordYToRow(float32 _fCoordY) const
{
	if (_fCoordY < m_fWindowMinY) return -1;
	if (_fCoordY > m_fWindowMaxY) return -1;
	
	int iRow = int((m_fWindowMaxY - _fCoordY) * m_fDivPixelLengthY);
	ASTRA_ASSERT(iRow >= 0);
	ASTRA_ASSERT(iRow < m_iGridRowCount);

	return iRow;
}

//----------------------------------------------------------------------------------------
// Convert a Z-coordinate to a slice index in the volume grid
inline int CVolumeGeometry3D::coordZToSlice(float32 _fCoordZ) const
{
	if (_fCoordZ < m_fWindowMinZ) return -1;
	if (_fCoordZ > m_fWindowMaxZ) return -1;
	
	int iSlice = int((m_fWindowMaxZ - _fCoordZ) * m_fDivPixelLengthZ);
	ASTRA_ASSERT(iSlice >= 0);
	ASTRA_ASSERT(iSlice < m_iGridSliceCount);

	return iSlice;
}

//----------------------------------------------------------------------------------------
// Convert an X-coordinate to a column index in the volume grid
inline float32 CVolumeGeometry3D::coordXToColFloat(float32 _fCoordX) const
{
	ASTRA_ASSERT(m_bInitialized);
	return (_fCoordX - m_fWindowMinX) * m_fDivPixelLengthX;
}

//----------------------------------------------------------------------------------------
// Convert a Y-coordinate to a row index in the volume grid
inline float32 CVolumeGeometry3D::coordYToRowFloat(float32 _fCoordY) const
{
	ASTRA_ASSERT(m_bInitialized);
	return (m_fWindowMaxY - _fCoordY) * m_fDivPixelLengthY;
}

//----------------------------------------------------------------------------------------
// Convert a Z-coordinate to a slice index in the volume grid
inline float32 CVolumeGeometry3D::coordZToSliceFloat(float32 _fCoordZ) const
{
	ASTRA_ASSERT(m_bInitialized);
	return (m_fWindowMaxZ - _fCoordZ) * m_fDivPixelLengthZ;
}
//----------------------------------------------------------------------------------------

} // end namespace astra

#endif /* _INC_ASTRA_VOLUMEGEOMETRY2D */
