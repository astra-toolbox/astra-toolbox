/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

#include <astra/Logger.h>

using namespace astra;

const char * g_loggerFileName = "astra_logger.txt";

void CLogger::_assureIsInitialized()
{
	if(!m_bInitialized)
	{
		m_pOutFile = fopen(g_loggerFileName, "r");
		if(m_pOutFile != NULL)
		{
			// file exists, users wants to log
			fclose(m_pOutFile);
			m_pOutFile = fopen(g_loggerFileName, "w");
		}

		m_bInitialized = true;
	}
}

void CLogger::writeLine(const char * _text)
{
	_assureIsInitialized();

	if(m_pOutFile != NULL)
	{
		fprintf(m_pOutFile, "%s\n", _text);
		fflush(m_pOutFile);
	}
}

void CLogger::writeTerminalCUDAError(const char * _fileName, int _iLine, const char * _errString)
{
	char buffer[256];

	sprintf(buffer, "Cuda error in file '%s' in line %i : %s.", _fileName, _iLine, _errString);

	writeLine(buffer);
}

CLogger::CLogger()
{
	;
}

FILE * CLogger::m_pOutFile = NULL;
bool CLogger::m_bInitialized = false;
