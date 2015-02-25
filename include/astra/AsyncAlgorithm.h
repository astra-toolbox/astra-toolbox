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

#ifndef _INC_ASTRA_ASYNCALGORITHM
#define _INC_ASTRA_ASYNCALGORITHM

#include "Config.h"
#include "Algorithm.h"

#ifdef USE_PTHREADS
#include <pthread.h>
#else
#include <boost/thread.hpp>
#endif

namespace astra {
	
/**
 * \brief
 * This class contains an wrapper algorithm that allows termination of its wrapped algorithm.  
 *
 * This is used to allow algorithm termination from matlab command line.
 */

class _AstraExport CAsyncAlgorithm : public CAlgorithm {
public:
	/** Default constructor, containing no code. 
	 */
	CAsyncAlgorithm();
	
	/** Constructor. 
	 */
	explicit CAsyncAlgorithm(CAlgorithm* _pAlg); 
	
	/** Destructor. 
	 */
	virtual ~CAsyncAlgorithm();

	/** Initialize using config object. 
	 */
	virtual bool initialize(const Config& _cfg);
	
	/** Initialize using algorithm pointer. 
	 */
	virtual bool initialize(CAlgorithm* _pAlg);

	/** Run the algorithm. 
	 */
	virtual void run(int _iNrIterations = 0);

	/** Return pointer to the wrapped algorithm. 
	 */
	CAlgorithm* getWrappedAlgorithm() { return m_pAlg; }

	/** Is the wrapped algorithm done. 
	 */
	bool isDone() const { return m_bDone; }	

	/** Signal abort to the wrapped algorithm. 
	 */
	void signalAbort();

protected:
	//< Has this class been initialized?
	bool m_bInitialized;
	
	//< Should wrapped algorithm be deleted after completion?
	bool m_bAutoFree;
	
	//< Pointer to wrapped algorithm.
	CAlgorithm* m_pAlg;
	
	//< Is the wrapped algorithm done. 
	volatile bool m_bDone;
	
#ifndef USE_PTHREADS
	//< Handle to boost thread object running the wrapped algorithm. 
	boost::thread* m_pThread;
#else
	pthread_t m_thread;
	struct AsyncThreadInfo {
		int m_iIterations;
		CAlgorithm* m_pAlg;
		volatile bool* m_pDone;
	} m_ThreadInfo;
	friend void* runAsync_pthreads(void*);
#endif
	bool m_bThreadStarted;

	//< Run the wrapped algorithm.
	void runWrapped(int _iNrIterations);

};

}

#endif
