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

#ifndef _INC_ASTRA_ASTRAOBJECTMANAGER
#define _INC_ASTRA_ASTRAOBJECTMANAGER

#include <map>
#include <sstream>

#include "Globals.h"
#include "Singleton.h"
#include "Projector2D.h"
#include "Projector3D.h"
#include "Float32Data2D.h"
#include "Float32Data3D.h"
#include "SparseMatrix.h"
#include "Algorithm.h"

namespace astra {

/**
 * This class contains functionality to store objects.  A unique index handle
 * will be assigned to each data object by which it can be accessed in the
 * future.  Indices are always >= 1.
 *
 * We store them in a special common base class to make indices unique
 * among all ObjectManagers.
 */

class CAstraObjectManagerBase {
public:
	virtual std::string getInfo(int index) const =0;
	virtual void remove(int index) =0;
	virtual std::string getType() const =0;
};


class _AstraExport CAstraIndexManager : public Singleton<CAstraIndexManager> {
public:
	CAstraIndexManager() : m_iLastIndex(0) { }

	int store(CAstraObjectManagerBase* m) {
		m_table[++m_iLastIndex] = m;
		return m_iLastIndex;
	}

	CAstraObjectManagerBase* get(int index) const {
		std::map<int, CAstraObjectManagerBase*>::const_iterator i;
		i = m_table.find(index);
		if (i != m_table.end())
			return i->second;
		else
			return 0;
	}

	void remove(int index) {
		std::map<int, CAstraObjectManagerBase*>::iterator i;
		i = m_table.find(index);
		if (i != m_table.end())
			m_table.erase(i);
	}

private:
	/** The index last handed out
	 */
	int m_iLastIndex;
	std::map<int, CAstraObjectManagerBase*> m_table;
};


template <typename T>
class CAstraObjectManager : public CAstraObjectManagerBase {

public:

	/** Default constructor.
	 */
	CAstraObjectManager();

	/** Destructor.  
	 */
	~CAstraObjectManager();

	/** Store the object in the manager and assign a unique index handle to it.
	 *
	 * @param _pObject A pointer to the object that should be stored.
	 * @return The index of the stored data object.  If the index in negative, an error occurred 
	 * and the object was NOT stored.
	 */
	int store(T* _pObject);

	/** Does the manager contain an object with the index _iIndex?
	 *
	 * @param _iIndex Index handle to the data object in question.
	 * @return True if the manager contains an object with the index handle _iIndex.
	 */
	bool hasIndex(int _iIndex) const;

	/** Fetch the object to which _iIndex refers to.
	 *
	 * @param _iIndex Index handle to the data object in question.
	 * @return Pointer to the stored data object.  A null pointer is returned if no object with index _iIndex is found.
	 */
	T* get(int _iIndex) const;

	/** Delete an object that was previously stored.  This actually DELETES the objecy.  Therefore, after this 
	* function call, the object in question will have passed on. It will be no more. It will have ceased 
	* to be.  It will be expired and will go to meet its maker.  Bereft of life, it will rest in peace.  
	* It will be an EX-OBJECT.
	*
	* @param _iIndex Index handle to the object in question.
	* @return Error code. 0 for success.
	*/
	void remove(int _iIndex);

	/** Get the index of the object, zero if it doesn't exist.
	 *
	 * @param _pObject The data object.
	 * @return Index of the stored object, 0 if not found.
	 */
	int getIndex(const T* _pObject) const;

	/** Clear all data.  This will also delete all the content of each object.
	 */
	void clear();

	/** Get info of object.
	 */
	std::string getInfo(int index) const;

	/** Get list with info of all managed objects.
	 */
	std::string info();

protected:

	/** Map each data object to a unique index.
	 */
	std::map<int, T*> m_mIndexToObject;

};

//----------------------------------------------------------------------------------------
// Constructor
template <typename T>
CAstraObjectManager<T>::CAstraObjectManager()
{
}

//----------------------------------------------------------------------------------------
// Destructor
template <typename T>
CAstraObjectManager<T>::~CAstraObjectManager()
{

}

//----------------------------------------------------------------------------------------
// store data
template <typename T>
int CAstraObjectManager<T>::store(T* _pDataObject) 
{
	int iIndex = CAstraIndexManager::getSingleton().store(this);
	m_mIndexToObject[iIndex] = _pDataObject;
	return iIndex;
}

//----------------------------------------------------------------------------------------
// has data?
template <typename T>
bool CAstraObjectManager<T>::hasIndex(int _iIndex) const
{
	typename map<int,T*>::const_iterator it = m_mIndexToObject.find(_iIndex);
	return it != m_mIndexToObject.end();
}

//----------------------------------------------------------------------------------------
// get data
template <typename T>
T* CAstraObjectManager<T>::get(int _iIndex) const
{
	typename map<int,T*>::const_iterator it = m_mIndexToObject.find(_iIndex);
	if (it != m_mIndexToObject.end())
		return it->second;
	else
		return 0;
}

//----------------------------------------------------------------------------------------
// delete data
template <typename T>
void CAstraObjectManager<T>::remove(int _iIndex)
{
	// find data
	typename map<int,T*>::iterator it = m_mIndexToObject.find(_iIndex);
	if (it == m_mIndexToObject.end())
		return;
	// delete data
	delete (*it).second;
	// delete from map
	m_mIndexToObject.erase(it);

	CAstraIndexManager::getSingleton().remove(_iIndex);
}

//----------------------------------------------------------------------------------------
// Get Index
template <typename T>
int CAstraObjectManager<T>::getIndex(const T* _pObject) const
{
	for (typename map<int,T*>::const_iterator it = m_mIndexToObject.begin(); it != m_mIndexToObject.end(); it++) {
		if ((*it).second == _pObject) return (*it).first;
	}
	return 0;
}


//----------------------------------------------------------------------------------------
// clear
template <typename T>
void CAstraObjectManager<T>::clear()
{
	for (typename map<int,T*>::iterator it = m_mIndexToObject.begin(); it != m_mIndexToObject.end(); it++) {
		// delete data
		delete (*it).second;
		(*it).second = 0;
	}

	m_mIndexToObject.clear();
}

//----------------------------------------------------------------------------------------
// Print info to string
template <typename T>
std::string CAstraObjectManager<T>::getInfo(int index) const {
	typename map<int,T*>::const_iterator it = m_mIndexToObject.find(index);
	if (it == m_mIndexToObject.end())
		return "";
	const T* pObject = it->second;
	std::stringstream res;
	res << index << " \t";
	if (pObject->isInitialized()) {
		res << "v     ";
	} else {
		res << "x     ";
	}
	res << pObject->description();
	return res.str();
}

template <typename T>
std::string CAstraObjectManager<T>::info() {
	std::stringstream res;
	res << "id  init  description" << std::endl;
	res << "-----------------------------------------" << std::endl;
	for (typename map<int,T*>::const_iterator it = m_mIndexToObject.begin(); it != m_mIndexToObject.end(); it++) {
		res << getInfo(it->first) << endl;
	}
	res << "-----------------------------------------" << std::endl;
	return res.str();
}



//----------------------------------------------------------------------------------------
// Create the necessary Object Managers
/**
 * This class contains functionality to store 2D projector objects.  A unique index handle will be 
 * assigned to each data object by which it can be accessed in the future.
 * Indices are always >= 1.
 */
class _AstraExport CProjector2DManager : public Singleton<CProjector2DManager>, public CAstraObjectManager<CProjector2D>
{
	virtual std::string getType() const { return "projector2d"; }
};

/**
 * This class contains functionality to store 3D projector objects.  A unique index handle will be 
 * assigned to each data object by which it can be accessed in the future.
 * Indices are always >= 1.
 */
class _AstraExport CProjector3DManager : public Singleton<CProjector3DManager>, public CAstraObjectManager<CProjector3D>
{
	virtual std::string getType() const { return "projector3d"; }
};

/**
 * This class contains functionality to store 2D data objects.  A unique index handle will be 
 * assigned to each data object by which it can be accessed in the future.
 * Indices are always >= 1.
 */
class _AstraExport CData2DManager : public Singleton<CData2DManager>, public CAstraObjectManager<CFloat32Data2D>
{
	virtual std::string getType() const { return "data2d"; }
};

/**
 * This class contains functionality to store 3D data objects.  A unique index handle will be 
 * assigned to each data object by which it can be accessed in the future.
 * Indices are always >= 1.
 */
class _AstraExport CData3DManager : public Singleton<CData3DManager>, public CAstraObjectManager<CFloat32Data3D>
{
	virtual std::string getType() const { return "data3d"; }
};

/**
 * This class contains functionality to store algorithm objects.  A unique index handle will be 
 * assigned to each data object by which it can be accessed in the future.
 * Indices are always >= 1.
 */
class _AstraExport CAlgorithmManager : public Singleton<CAlgorithmManager>, public CAstraObjectManager<CAlgorithm>
{
	virtual std::string getType() const { return "algorithm"; }
};

/**
 * This class contains functionality to store matrix objects.  A unique index handle will be 
 * assigned to each data object by which it can be accessed in the future.
 * Indices are always >= 1.
 */
class _AstraExport CMatrixManager : public Singleton<CMatrixManager>, public CAstraObjectManager<CSparseMatrix>
{
	virtual std::string getType() const { return "matrix"; }
};


} // end namespace

#endif
