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

#ifndef _INC_ASTRA_DATAPROJECTOR
#define _INC_ASTRA_DATAPROJECTOR

#include "Projector2D.h"

#include "TypeList.h"

#include "ProjectorTypelist.h"

#include "DataProjectorPolicies.h"

namespace astra
{


/**
 * Interface class for the Data Projector. The sole purpose of this class is to force child classes to implement a series of methods
 */
class CDataProjectorInterface {
public:
	CDataProjectorInterface() { }
	virtual ~CDataProjectorInterface() { }
	virtual void project() = 0;
	virtual void projectSingleProjection(int _iProjection) = 0;
	virtual void projectSingleRay(int _iProjection, int _iDetector) = 0;
//	virtual void projectSingleVoxel(int _iRow, int _iCol) = 0;
//	virtual void projectAllVoxels() = 0;
};

/**
 * Templated Data Projector Class. In this class a specific projector and policies are combined.
 */
template <typename Projector, typename Policy>
class CDataProjector: public CDataProjectorInterface {

private:

	Projector* m_pProjector;
	Policy m_pPolicy;

public:

	CDataProjector() {};

	CDataProjector(Projector* _p, Policy _a);
	~CDataProjector();

	virtual void project();

	virtual void projectSingleProjection(int _iProjection);

	virtual void projectSingleRay(int _iProjection, int _iDetector);

//	virtual void projectSingleVoxel(int _iRow, int _iCol);

//	virtual void projectAllVoxels();
};

//----------------------------------------------------------------------------------------
/**
 * Constructor
*/
template <typename Projector, typename Policy>
CDataProjector<Projector,Policy>::CDataProjector(Projector* _p, Policy _a) 
{ 
	m_pProjector = _p;
	m_pPolicy = _a; 
}

//----------------------------------------------------------------------------------------
/**
 * Destructor
*/
template <typename Projector, typename Policy>
CDataProjector<Projector,Policy>::~CDataProjector() 
{ 
	// does nothing
}

//----------------------------------------------------------------------------------------
/**
 * Compute projection using the algorithm specific to the projector type
*/
template <typename Projector, typename Policy>
void CDataProjector<Projector,Policy>::project() 
{ 
	m_pProjector->project(m_pPolicy);
}

//----------------------------------------------------------------------------------------
/**
 * Compute just one projection using the algorithm specific to the projector type
*/
template <typename Projector, typename Policy>
void CDataProjector<Projector,Policy>::projectSingleProjection(int _iProjection) 
{ 
	m_pProjector->projectSingleProjection(_iProjection, m_pPolicy);
}

//----------------------------------------------------------------------------------------
/**
 * Compute projection of one ray using the algorithm specific to the projector type
*/
template <typename Projector, typename Policy>
void CDataProjector<Projector,Policy>::projectSingleRay(int _iProjection, int _iDetector)
{ 
	m_pProjector->projectSingleRay(_iProjection, _iDetector, m_pPolicy);
}

//----------------------------------------------------------------------------------------
//template <typename Projector, typename Policy>
//void CDataProjector<Projector,Policy>::projectSingleVoxel(int _iRow, int _iCol) 
//{ 
//	m_pProjector->projectSingleVoxel(_iRow, _iCol, m_pPolicy);
//}

//----------------------------------------------------------------------------------------
//template <typename Projector, typename Policy>
//void CDataProjector<Projector,Policy>::projectAllVoxels() 
//{ 
//	m_pProjector->projectAllVoxels(m_pPolicy);
//}
//----------------------------------------------------------------------------------------




//-----------------------------------------------------------------------------------------
// Create a new datainterface from the projector TypeList
namespace typelist {
	template <class TList> 
	struct CreateDataProjector { 
		template <class U, typename Policy>
		 static void find (U& functor, CProjector2D* _pProjector, const Policy& _pPolicy) {
			 if (functor(TList::Head::type)) {
				functor.res = new CDataProjector<typename TList::Head, Policy>(static_cast<typename TList::Head*>(_pProjector), _pPolicy);
			 }
			 CreateDataProjector<typename TList::Tail>::find(functor, _pProjector, _pPolicy); 
		 }
	}; 
	template <> 
	struct CreateDataProjector<NullType> {
		template <class U, typename Policy> 
		static void find(U& functor, CProjector2D* _pProjector, const Policy& _pPolicy) {}
	}; 

	struct functor_find_datainterface {
		functor_find_datainterface() { res = NULL; }
		bool operator() (std::string name) { 
			return strcmp(tofind.c_str(), name.c_str()) == 0;
		} 
		std::string tofind;
		CDataProjectorInterface* res;
	};
}
//-----------------------------------------------------------------------------------------

/**
 * Data Projector Dispatcher - 1 Policy
 */
template <typename Policy>
static CDataProjectorInterface* dispatchDataProjector(CProjector2D* _pProjector, const Policy& _policy)
{
	typelist::functor_find_datainterface finder = typelist::functor_find_datainterface();
	finder.tofind = _pProjector->getType();
	typelist::CreateDataProjector<Projector2DTypeList>::find(finder, _pProjector, _policy);
	return finder.res;
}



/**
 * Data Projector Dispatcher - 2 Policies
 */
template <typename Policy1, typename Policy2>
static CDataProjectorInterface* dispatchDataProjector(CProjector2D* _pProjector, 
													  const Policy1& _policy,
													  const Policy2& _policy2,
													  bool _bUsePolicy1 = true, 
													  bool _bUsePolicy2 = true) 
{
	if (!_bUsePolicy1 && !_bUsePolicy2) {
		return dispatchDataProjector(_pProjector, EmptyPolicy());
	} else if (!_bUsePolicy1) {
		return dispatchDataProjector(_pProjector, _policy2);
	} else if (!_bUsePolicy2) {
		return dispatchDataProjector(_pProjector, _policy);
	} else {
		return dispatchDataProjector(_pProjector, CombinePolicy<Policy1, Policy2>(_policy, _policy2));
	}
	
}

/**
 * Data Projector Dispatcher - 3 Policies
 */

template <typename Policy1, typename Policy2, typename Policy3>
static CDataProjectorInterface* dispatchDataProjector(CProjector2D* _pProjector, 
													  const Policy1& _policy1,
													  const Policy2& _policy2,
													  const Policy3& _policy3,
													  bool _bUsePolicy1 = true, 
													  bool _bUsePolicy2 = true,
													  bool _bUsePolicy3 = true) 
{
	if (!_bUsePolicy1) {
		return dispatchDataProjector(_pProjector, _policy2, _policy3, _bUsePolicy2, _bUsePolicy3);
	} else if (!_bUsePolicy2) {
		return dispatchDataProjector(_pProjector, _policy1, _policy3, _bUsePolicy1, _bUsePolicy3);
	} else if (!_bUsePolicy3) {
		return dispatchDataProjector(_pProjector, _policy1, _policy2, _bUsePolicy1, _bUsePolicy2);
	} else {
		return dispatchDataProjector(_pProjector, Combine3Policy<Policy1, Policy2, Policy3>(_policy1, _policy2, _policy3));
	}
}

/**
 * Data Projector Dispatcher - 4 Policies
 */
template <typename Policy1, typename Policy2, typename Policy3, typename Policy4>
static CDataProjectorInterface* dispatchDataProjector(CProjector2D* _pProjector, 
													  const Policy1& _policy1,
													  const Policy2& _policy2,
													  const Policy3& _policy3,
													  const Policy4& _policy4,
													  bool _bUsePolicy1 = true, 
													  bool _bUsePolicy2 = true,
													  bool _bUsePolicy3 = true,
													  bool _bUsePolicy4 = true) 
{
	if (!_bUsePolicy1) {
		return dispatchDataProjector(_pProjector, _policy2, _policy3, _policy4, _bUsePolicy2, _bUsePolicy3, _bUsePolicy4);
	} else if (!_bUsePolicy2) {
		return dispatchDataProjector(_pProjector, _policy1, _policy3, _policy4, _bUsePolicy1, _bUsePolicy3, _bUsePolicy4);
	} else if (!_bUsePolicy3) {
		return dispatchDataProjector(_pProjector, _policy1, _policy2, _policy4, _bUsePolicy1, _bUsePolicy2, _bUsePolicy4);
	} else if (!_bUsePolicy4) {
		return dispatchDataProjector(_pProjector, _policy1, _policy2, _policy3, _bUsePolicy1, _bUsePolicy2, _bUsePolicy3);
	} else {
		return dispatchDataProjector(_pProjector, Combine4Policy<Policy1, Policy2, Policy3, Policy4>(_policy1, _policy2, _policy3, _policy4));
	}
}

/**
 * Data Projector Dispatcher - 5 Policies
 */
template <typename Policy1, typename Policy2, typename Policy3, typename Policy4, typename Policy5>
static CDataProjectorInterface* dispatchDataProjector(CProjector2D* _pProjector, 
													  const Policy1& _policy1,
													  const Policy2& _policy2,
													  const Policy3& _policy3,
													  const Policy4& _policy4,
													  const Policy5& _policy5,
													  bool _bUsePolicy1 = true, 
													  bool _bUsePolicy2 = true,
													  bool _bUsePolicy3 = true,
													  bool _bUsePolicy4 = true,
													  bool _bUsePolicy5 = true) 
{
	if (!_bUsePolicy1) {
		return dispatchDataProjector(_pProjector, _policy2, _policy3, _policy4, _policy5, _bUsePolicy2, _bUsePolicy3, _bUsePolicy4, _bUsePolicy5);
	} else if (!_bUsePolicy2) {
		return dispatchDataProjector(_pProjector, _policy1, _policy3, _policy4, _policy5, _bUsePolicy1, _bUsePolicy3, _bUsePolicy4, _bUsePolicy5);
	} else if (!_bUsePolicy3) {
		return dispatchDataProjector(_pProjector, _policy1, _policy2, _policy4, _policy5, _bUsePolicy1, _bUsePolicy2, _bUsePolicy4, _bUsePolicy5);
	} else if (!_bUsePolicy4) {
		return dispatchDataProjector(_pProjector, _policy1, _policy2, _policy3, _policy5, _bUsePolicy1, _bUsePolicy2, _bUsePolicy3, _bUsePolicy5);
	} else if (!_bUsePolicy5) {
		return dispatchDataProjector(_pProjector, _policy1, _policy2, _policy3, _policy4, _bUsePolicy1, _bUsePolicy2, _bUsePolicy3, _bUsePolicy4);
	} else {
		return dispatchDataProjector(_pProjector, CombinePolicy< Combine4Policy<Policy1, Policy2, Policy3, Policy4>, Policy5>(
														Combine4Policy<Policy1, Policy2, Policy3, Policy4>(_policy1, _policy2, _policy3, _policy4), 
														_policy5)
									);
	}
}




//-----------------------------------------------------------------------------------------
/**
 * Data Projector Project
 */
template <typename Policy>
static void projectData(CProjector2D* _pProjector, const Policy& _policy)
{
	CDataProjectorInterface* dp = dispatchDataProjector(_pProjector, _policy);
	dp->project();
	delete dp;
}




} // namespace astra

#endif 
