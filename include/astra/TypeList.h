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

#ifndef _INC_ASTRA_TYPELIST
#define _INC_ASTRA_TYPELIST

#include "Globals.h"
#include <cstring>

namespace astra {

template<class... Ts>
struct TypeList {

};

template<typename Base, class T, class... Ts>
Base* createObject_internal(const std::string &name)
{
	if (name == T::type)
		return new T();
	if constexpr (sizeof...(Ts) > 0)
		return createObject_internal<Base, Ts...>(name);
	else
		return nullptr;
}

template<typename Base, class... Ts>
Base* createObject(const std::string &name, TypeList<Ts...>)
{
	if constexpr (sizeof...(Ts) > 0)
		return createObject_internal<Base, Ts...>(name);
	else
		return nullptr;
}

} // end namespace astra

#endif
