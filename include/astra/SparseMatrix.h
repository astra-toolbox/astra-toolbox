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

#ifndef _INC_ASTRA_SPARSEMATRIX
#define _INC_ASTRA_SPARSEMATRIX

namespace astra
{


/** This class implements a sparse matrix. It is stored as three arrays.
 *  The values are stored row-by-row.
 *  m_pfValues     contains the values
 *  m_piColIndices contains the col indices of the values
 *  m_plRowStarts  contains the start offsets of the rows
 */


class _AstraExport CSparseMatrix {
public:
	CSparseMatrix();

	// TODO: are ints large enough for width/height?
	CSparseMatrix(unsigned int _iHeight, unsigned int _iWidth,
	              unsigned long _lSize);

	/** Initialize the matrix structure.
	 *  It does not initialize any values.
	 *
	 * @param _iHeight number of rows
	 * @param _iWidth number of columns
	 * @param _lSize maximum number of non-zero entries
	 * @return initialization successful?
	 */

	bool initialize(unsigned int _iHeight, unsigned int _iWidth,
	                unsigned long _lSize);

	/** Destructor.
	 */
	~CSparseMatrix();

	/** Has the matrix structure been initialized?
	 *
	 * @return initialized successfully
	 */
	bool isInitialized() const { return m_bInitialized; }

	/** get a description of the class
	 *
	 * @return description string
	 */
	std::string description() const;

	/** get the data for a single row. Entries are stored from left to right.
	 *
	 * @param _iRow the row
	 * @param _iSize the returned number of elements in the row
	 * @param _pfValues the values of the non-zero entries in the row
	 * @param _piColIndices the column indices of the non-zero entries
	 */
	void getRowData(unsigned int _iRow, unsigned int& _iSize,
	                   const float32*& _pfValues, const unsigned int*& _piColIndices) const
	{
		assert(_iRow < m_iHeight);
		unsigned long lStart = m_plRowStarts[_iRow];
		_iSize = m_plRowStarts[_iRow+1] - lStart;
		_pfValues = &m_pfValues[lStart];
		_piColIndices = &m_piColIndices[lStart];
	}

	/** get the number of elements in a row
	 *
	 * @param _iRow the row
	 * @return number of stored entries in the row
	 */
	unsigned int getRowSize(unsigned int _iRow) const
	{
		assert(_iRow < m_iHeight);
		return m_plRowStarts[_iRow+1] - m_plRowStarts[_iRow];
	}


	/** Matrix width
	 */
	unsigned int m_iHeight;
	
	/** Matrix height
	 */
	unsigned int m_iWidth;
	
	/** Maximum number of non-zero entries
	 */
	unsigned long m_lSize;

	/** Contains the numeric values of all non-zero elements
	 */
	float32* m_pfValues;
	
	/** Contains the colon index of all non-zero elements
	 */
	unsigned int* m_piColIndices;
	
	/** The indices in this array point to the first element of each row in the m_pfValues array
	 */
	unsigned long* m_plRowStarts;

protected:
	
	/** Is the class initialized?
	 */
	bool m_bInitialized;
};


}


#endif
