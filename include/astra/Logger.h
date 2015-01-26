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

#ifndef _INC_ASTRA_LOGGER
#define _INC_ASTRA_LOGGER

#include <cstdio>

namespace astra
{

/**
 *  This is the first stab at a decent logger. If the file "astra_logger.txt", it will be replaced
 *  with the text sent to this logger. If the file doesn't exist when the app starts, nothing is written.
 */
class CLogger
{
	static std::FILE * m_pOutFile;
	static bool m_bInitialized;

	static void _assureIsInitialized();

	CLogger();

public:

	/**
	 * Writes a line to the log file (newline is added). Ignored if logging is turned off.
	 *
	 * @param _text char pointer to text in line
	 */
	static void writeLine(const char * _text);

	/**
	 * Formats and writes a CUDA error to the log file. Ignored if logging is turned off.
	 *
	 * @param _fileName filename where error occurred (typically __FILE__)
	 * @param _line line in file (typically __LINE__)
	 * @param _errString string describing the error, can be output of cudaGetErrorString
	 */
	static void writeTerminalCUDAError(const char * _fileName, int _iLine, const char * _errString);
};

}

#endif /* _INC_ASTRA_LOGGER */

