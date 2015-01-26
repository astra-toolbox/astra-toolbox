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

#ifndef _INC_ASTRA_SINGLETON
#define _INC_ASTRA_SINGLETON

#include <cassert>

#ifndef _MSC_VER
#include <stdint.h>
#endif

namespace astra {
	/** 
	 * This singleton interface class ensures that any of its children can be instatiated only once. This is used by the ObjectFactories.
	 **/
template<typename T>
class Singleton {

	public:

		// constructor
		Singleton() {
			assert(!m_singleton);
			int offset = (uintptr_t)(T*)1 - (uintptr_t)(Singleton<T>*)(T*)1;
			m_singleton = (T*)((uintptr_t)this + offset);
		};

		// destructor
		virtual ~Singleton() {
			assert(m_singleton);
			m_singleton = 0;
		}

		// get singleton
		static T& getSingleton() {
			if (!m_singleton)
				m_singleton = new T();
			return *m_singleton;
		}
		static T* getSingletonPtr() {
			if (!m_singleton)
				m_singleton = new T();
			return m_singleton;
		}

	private:

		// the singleton
		static T* m_singleton;

};

#define DEFINE_SINGLETON(T) template<> T* Singleton<T >::m_singleton = 0

// This is a hack to support statements like
// DEFINE_SINGLETON2(CTemplatedClass<C1, C2>);
#define DEFINE_SINGLETON2(A,B) template<> A,B* Singleton<A,B >::m_singleton = 0

} // end namespace

#endif
