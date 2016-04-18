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

#ifndef _INC_ASTRA_ALGORITHM
#define _INC_ASTRA_ALGORITHM

#include <boost/any.hpp>

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

	/** Get all information parameters
	 *
	 * @return map with all boost::any object
	 */
	virtual map<string,boost::any> getInformation();

	/** Get a single piece of information represented as a boost::any
	 *
	 * @param _sIdentifier identifier string to specify which piece of information you want
	 * @return boost::any object
	 */
	virtual boost::any getInformation(std::string _sIdentifier);

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

	/** Signal the algorithm it should abort soon.
	 *  This is intended to be called from a different thread
	 *  while the algorithm is running. There are no guarantees
	 *  on how soon the algorithm will abort. The state of the
	 *  algorithm object will be consistent (so it is safe to delete it
	 *  normally afterwards), but the algorithm's output is undefined.
	 *
	 *  Note that specific algorithms may give guarantees on their
	 *  state after an abort. Check their documentation for details.
	 */
	virtual void signalAbort() { m_bShouldAbort = true; }

protected:

	//< Has this class been initialized?
	bool m_bIsInitialized;

	//< If this is set, the algorithm should try to abort as soon as possible.
	volatile bool m_bShouldAbort;

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
