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

#ifndef PLATFORMDEPSYSTEMCODE_H
#define PLATFORMDEPSYSTEMCODE_H

#include <cstdio>

#ifndef _WIN32
#include <stdint.h>
#endif

namespace astra
{

#ifdef _WIN32
	typedef __int64 int64;
#else
	typedef int64_t int64;
#endif

class CPlatformDepSystemCode
{
public:
	
	/**
	 * Clock with resolution of 1 ms. Windows implementation will return number of ms since system start,
	 * but this is not a requirement for the implementation. Just as long as the subtraction of two acquired
	 * values will result in a time interval in ms.
	 *
	 * @return a value that increases with 1 every ms
	 */
	static unsigned long getMSCount();

	/**
	 * fseek variant that works with 64 bit ints. 
	 *
	 * @param _pStream file handler of file in which needs to be seek-ed
	 * @param _iOffset 64 bit int telling the new offset in the file
	 * @param _iOrigin typical fseek directive telling how _iOffset needs to be interpreted (SEEK_SET, ...)
	 *
	 * @return 0 if successful
	 */
	static int fseek64(FILE * _pStream, astra::int64 _iOffset, int _iOrigin);

	/**
	 * 64-bit ftell variant
	 *
	 * @param _pStream file handle
	 *
	 * @return the position in the file
	 */
	static astra::int64 ftell64(FILE * _pStream);
};

}

#endif /* PLATFORMDEPSYSTEMCODE_H */
