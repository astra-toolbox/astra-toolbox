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

#ifndef _INC_ASTRA_ASTRAOBJECTFACTORY
#define _INC_ASTRA_ASTRAOBJECTFACTORY

#include "Globals.h"
#include "Config.h"
#include "Singleton.h"
#include "Utilities.h"
#include "TypeList.h"

#include "ProjectorTypelist.h"


#include "AlgorithmTypelist.h"

#ifdef ASTRA_PYTHON
#include "PluginAlgorithm.h"
#endif


namespace astra {

/**
 * This class contains functionality to create data objects based on their type or on a configuration object.
 */
template <typename T, typename TypeList>
class CAstraObjectFactory : public Singleton<CAstraObjectFactory<T, TypeList> > {

public:

	/** A default constructor that contains not a single line of code.
	 */
	CAstraObjectFactory();

	/** Destructor.  
	 */
	~CAstraObjectFactory();

	/** Create, but don't initialize, a new object.
	 *
	 * @param _sType Type of the new object.
	 * @return Pointer to a new, uninitialized object.
	 */
	T* create(std::string _sType);

	/** Create and initialize a new object.
	 *
	 * @param _cfg Configuration object to create and initialize a new object.
	 * @return Pointer to a new, initialized projector.
	 */
	T* create(const Config& _cfg);

	/** Find a plugin.
	*
	* @param _sType Name of plugin to find.
	* @return Pointer to a new, uninitialized object, or NULL if not found.
	*/
	T* findPlugin(std::string _sType);


};


//----------------------------------------------------------------------------------------
// Constructor
template <typename T, typename TypeList>
CAstraObjectFactory<T, TypeList>::CAstraObjectFactory()
{

}

//----------------------------------------------------------------------------------------
// Destructor
template <typename T, typename TypeList>
CAstraObjectFactory<T, TypeList>::~CAstraObjectFactory()
{

}


//----------------------------------------------------------------------------------------
// Hook for finding plugin in registered plugins.
template <typename T, typename TypeList>
T* CAstraObjectFactory<T, TypeList>::findPlugin(std::string _sType)
{
	return NULL;
}

//----------------------------------------------------------------------------------------
// Create 
template <typename T, typename TypeList>
T* CAstraObjectFactory<T, TypeList>::create(std::string _sType) 
{
	functor_find<T> finder = functor_find<T>();
	finder.tofind = _sType;
	CreateObject<TypeList>::find(finder);
	if (finder.res == NULL) {
		finder.res = findPlugin(_sType);
	}
	return finder.res;
}

//----------------------------------------------------------------------------------------
// Create with XML
template <typename T, typename TypeList>
T* CAstraObjectFactory<T, TypeList>::create(const Config& _cfg)
{
	T* object = create(_cfg.self.getAttribute("type"));
	if (object == NULL) return NULL;
	if (object->initialize(_cfg))
		return object;
	delete object;
	return NULL;
}
//----------------------------------------------------------------------------------------




//----------------------------------------------------------------------------------------
// Create the necessary Object Managers
/**
 * Class used to create algorithms from a string or a config object
*/
class _AstraExport CAlgorithmFactory : public CAstraObjectFactory<CAlgorithm, AlgorithmTypeList> {};

#ifdef ASTRA_PYTHON
template <>
inline CAlgorithm* CAstraObjectFactory<CAlgorithm, AlgorithmTypeList>::findPlugin(std::string _sType)
	{
		CPluginAlgorithmFactory *fac = CPluginAlgorithmFactory::getSingletonPtr();
		return fac->getPlugin(_sType);
	}
#endif

/**
 * Class used to create 2D projectors from a string or a config object
*/
class _AstraExport CProjector2DFactory : public CAstraObjectFactory<CProjector2D, Projector2DTypeList> {};

/**
 * Class used to create 3D projectors from a string or a config object
*/
class _AstraExport CProjector3DFactory : public CAstraObjectFactory<CProjector3D, Projector3DTypeList> {};




} // end namespace

#endif
