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

#ifndef _INC_ASTRA_ALGORITHM
#define _INC_ASTRA_ALGORITHM

#include "Globals.h"
#include "Config.h"

namespace astra {

/**
 * This class contains the interface for an algorithm implementation.
 */
class _AstraExport CAlgorithm {

public:
	
	/** Default constructor, containing no code.
	 */
	CAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CAlgorithm();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg) = 0;

	/** Perform a number of iterations.
	 *
	 * @param _iNrIterations amount of iterations to perform.
	 */
	virtual void run(int _iNrIterations = 0) = 0;

	/** Has this class been initialized?
	 *
	 * @return initialized
	 */
	bool isInitialized() const;

	/** get a description of the class
	 *
	 * @return description string
	 */
	virtual std::string description() const;

	/** Set the GPU Index to run on.
	 * TODO: Move this from CAlgorithm to a Context-like class
	 */
	virtual void setGPUIndex(int /*_iGPUIndex*/) { };

protected:

	//< Has this class been initialized?
	bool m_bIsInitialized;

private:
	/**
	 * Private copy constructor to prevent CAlgorithms from being copied.
	 */
	CAlgorithm(const CAlgorithm&);

	/**
	 * Private assignment operator to prevent CAlgorithms from being copied.
	 */
	CAlgorithm& operator=(const CAlgorithm&);

	//< For Config unused argument checking
	ConfigCheckData* configCheckData;
	friend class ConfigStackCheck<CAlgorithm>;

};

// inline functions
inline std::string CAlgorithm::description() const { return "Algorithm"; };
inline bool CAlgorithm::isInitialized() const { return m_bIsInitialized; }

} // end namespace

#endif
