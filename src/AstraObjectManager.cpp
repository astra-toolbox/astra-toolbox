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

#include "astra/AstraObjectManager.h"


namespace astra {


CAstraIndexManager::CAstraIndexManager()
    : m_iLastIndex(0)
{

}

// Explicit destructor to prevent a potentially inlined implicit destructor
// destructing our mutex (which could happen in a different DLL in Windows)
CAstraIndexManager::~CAstraIndexManager()
{

}

int CAstraIndexManager::store(CAstraObjectManagerBase* m) {
	std::unique_lock lock{m_mutex};

	m_table[++m_iLastIndex] = m;
	return m_iLastIndex;
}

CAstraObjectManagerBase* CAstraIndexManager::get(int index) const {
	std::unique_lock lock{m_mutex};

	std::map<int, CAstraObjectManagerBase*>::const_iterator i;
	i = m_table.find(index);
	if (i != m_table.end())
		return i->second;
	else
		return 0;
}

void CAstraIndexManager::remove(int index) {
	std::unique_lock lock{m_mutex};

	std::map<int, CAstraObjectManagerBase*>::iterator i;
	i = m_table.find(index);
	if (i != m_table.end())
		m_table.erase(i);
}


template <typename T>
CAstraObjectManager<T>::CAstraObjectManager()
{
}

template <typename T>
CAstraObjectManager<T>::~CAstraObjectManager()
{

}

template <typename T>
int CAstraObjectManager<T>::store(T* _pDataObject)
{
	std::unique_lock lock{m_mutex};

	int iIndex = CAstraIndexManager::getSingleton().store(this);
	m_mIndexToObject[iIndex] = _pDataObject;
	return iIndex;
}

template <typename T>
bool CAstraObjectManager<T>::hasIndex(int _iIndex) const
{
	std::unique_lock lock{m_mutex};

	typename std::map<int,T*>::const_iterator it = m_mIndexToObject.find(_iIndex);
	return it != m_mIndexToObject.end();
}

template <typename T>
T* CAstraObjectManager<T>::get(int _iIndex) const
{
	std::unique_lock lock{m_mutex};

	typename std::map<int,T*>::const_iterator it = m_mIndexToObject.find(_iIndex);
	if (it != m_mIndexToObject.end())
		return it->second;
	else
		return 0;
}

template <typename T>
void CAstraObjectManager<T>::remove(int _iIndex)
{
	std::unique_lock lock{m_mutex};

	// find data
	typename std::map<int,T*>::iterator it = m_mIndexToObject.find(_iIndex);
	if (it == m_mIndexToObject.end())
		return;
	// delete data
	delete (*it).second;
	// delete from map
	m_mIndexToObject.erase(it);

	CAstraIndexManager::getSingleton().remove(_iIndex);
}

template <typename T>
int CAstraObjectManager<T>::getIndex(const T* _pObject) const
{
	std::unique_lock lock{m_mutex};

	for (typename std::map<int,T*>::const_iterator it = m_mIndexToObject.begin(); it != m_mIndexToObject.end(); it++) {
		if ((*it).second == _pObject) return (*it).first;
	}
	return 0;
}


template <typename T>
void CAstraObjectManager<T>::clear()
{
	std::unique_lock lock{m_mutex};

	for (typename std::map<int,T*>::iterator it = m_mIndexToObject.begin(); it != m_mIndexToObject.end(); it++) {
		// delete data
		delete (*it).second;
		(*it).second = 0;
	}

	m_mIndexToObject.clear();
}

template <typename T>
std::string CAstraObjectManager<T>::getInfo(int index) const {
	std::unique_lock lock{m_mutex};

	typename std::map<int,T*>::const_iterator it = m_mIndexToObject.find(index);
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
	std::unique_lock lock{m_mutex};

	std::stringstream res;
	res << "id  init  description" << std::endl;
	res << "-----------------------------------------" << std::endl;
	for (typename std::map<int,T*>::const_iterator it = m_mIndexToObject.begin(); it != m_mIndexToObject.end(); it++) {
		res << getInfo(it->first) << std::endl;
	}
	res << "-----------------------------------------" << std::endl;
	return res.str();
}



template class CAstraObjectManager<CProjector2D>;
template class CAstraObjectManager<CProjector3D>;
template class CAstraObjectManager<CData2D>;
template class CAstraObjectManager<CData3D>;
template class CAstraObjectManager<CAlgorithm>;
template class CAstraObjectManager<CSparseMatrix>;



DEFINE_SINGLETON(CProjector2DManager)
DEFINE_SINGLETON(CProjector3DManager)
DEFINE_SINGLETON(CData2DManager)
DEFINE_SINGLETON(CData3DManager)
DEFINE_SINGLETON(CAlgorithmManager)
DEFINE_SINGLETON(CMatrixManager)

DEFINE_SINGLETON(CAstraIndexManager)

} // end namespace
