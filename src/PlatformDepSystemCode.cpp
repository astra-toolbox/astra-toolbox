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

#include "astra/PlatformDepSystemCode.h"

using namespace astra;

#ifdef _WIN32
#include "windows.h"
// windows API available

unsigned long CPlatformDepSystemCode::getMSCount()
{
	return ::GetTickCount();
}

int CPlatformDepSystemCode::fseek64(FILE * _pStream, astra::int64 _iOffset, int _iOrigin)
{
	return _fseeki64(_pStream, _iOffset, _iOrigin);
}

astra::int64 CPlatformDepSystemCode::ftell64(FILE * _pStream)
{
	return _ftelli64(_pStream);
}

#else
// linux, ...

#include <sys/time.h>

unsigned long CPlatformDepSystemCode::getMSCount()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (tv.tv_sec * 1000) + (tv.tv_usec/1000);
}

int CPlatformDepSystemCode::fseek64(FILE * _pStream, astra::int64 _iOffset, int _iOrigin)
{
	return fseeko(_pStream, _iOffset, _iOrigin);
}

astra::int64 CPlatformDepSystemCode::ftell64(FILE * _pStream)
{
	return ftello(_pStream);
}



#endif
