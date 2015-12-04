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

#define CLOG_MAIN
#include <astra/clog.h>

#include <astra/Logging.h>

#include <cstdio>

using namespace astra;

void CLogger::enableScreen()
{
	m_bEnabledScreen = true;
}

void CLogger::enableFile()
{
	m_bEnabledFile = true;
}

void CLogger::enable()
{
	enableScreen();
	enableFile();
}

void CLogger::disableScreen()
{
	m_bEnabledScreen = false;
}

void CLogger::disableFile()
{
	m_bEnabledFile = false;
}

void CLogger::disable()
{
	disableScreen();
	disableFile();
}

void CLogger::debug(const char *sfile, int sline, const char *fmt, ...)
{
	_assureIsInitialized();
	va_list ap, apf;
	if(m_bEnabledScreen){
        va_start(ap, fmt);
        clog_debug(sfile,sline,0,fmt,ap);
        va_end(ap);
    }
	if(m_bEnabledFile && m_bFileProvided){
        va_start(apf, fmt);
        clog_debug(sfile,sline,1,fmt,apf);
        va_end(apf);
    }
}

void CLogger::info(const char *sfile, int sline, const char *fmt, ...)
{
	_assureIsInitialized();
	va_list ap, apf;
	if(m_bEnabledScreen){
        va_start(ap, fmt);
        clog_info(sfile,sline,0,fmt,ap);
        va_end(ap);
    }
	if(m_bEnabledFile && m_bFileProvided){
        va_start(apf, fmt);
        clog_info(sfile,sline,1,fmt,apf);
        va_end(apf);
    }
}

void CLogger::warn(const char *sfile, int sline, const char *fmt, ...)
{
	_assureIsInitialized();
	va_list ap, apf;
	if(m_bEnabledScreen){
        va_start(ap, fmt);
        clog_warn(sfile,sline,0,fmt,ap);
        va_end(ap);
    }
	if(m_bEnabledFile && m_bFileProvided){
        va_start(apf, fmt);
        clog_warn(sfile,sline,1,fmt,apf);
        va_end(apf);
    }
}

void CLogger::error(const char *sfile, int sline, const char *fmt, ...)
{
	_assureIsInitialized();
	va_list ap, apf;
	if(m_bEnabledScreen){
        va_start(ap, fmt);
        clog_error(sfile,sline,0,fmt,ap);
        va_end(ap);
    }
	if(m_bEnabledFile && m_bFileProvided){
        va_start(apf, fmt);
        clog_error(sfile,sline,1,fmt,apf);
        va_end(apf);
    }
}

void CLogger::_setLevel(int id, log_level m_eLevel)
{
	switch(m_eLevel){
		case LOG_DEBUG:
			clog_set_level(id,CLOG_DEBUG);
			break;
		case LOG_INFO:
			clog_set_level(id,CLOG_INFO);
			break;
		case LOG_WARN:
			clog_set_level(id,CLOG_WARN);
			break;
		case LOG_ERROR:
			clog_set_level(id,CLOG_ERROR);
			break;
	}
}

void CLogger::setOutputScreen(int fd, log_level m_eLevel)
{
	_assureIsInitialized();
	if(fd==1||fd==2){
		clog_set_fd(0, fd);
	}else{
		error(__FILE__,__LINE__,"Invalid file descriptor");
	}
	_setLevel(0,m_eLevel);
}

void CLogger::setOutputFile(const char *filename, log_level m_eLevel)
{
	if(m_bFileProvided){
		clog_free(1);
		m_bFileProvided=false;
	}
	if(!clog_init_path(1,filename)){
		m_bFileProvided=true;
		_setLevel(1,m_eLevel);
	}
}

void CLogger::_assureIsInitialized()
{
	if(!m_bInitialized)
	{
		clog_init_fd(0, 2);
		clog_set_level(0, CLOG_INFO);
		clog_set_fmt(0, "%l: %m\n");
		#if USE_MPI
   		  clog_set_fmt(0, "%D %l: %m\n");
		#endif
		m_bInitialized = true;
	}
}

void CLogger::setFormatFile(const char *fmt)
{
	if(m_bFileProvided){
		clog_set_fmt(1,fmt);
	}else{
		error(__FILE__,__LINE__,"No log file specified");
	}
}
void CLogger::setFormatScreen(const char *fmt)
{
	clog_set_fmt(0,fmt);
}

CLogger::CLogger()
{
	;
}

bool CLogger::setCallbackScreen(void (*cb)(const char *msg, size_t len)){
	_assureIsInitialized();
	return clog_set_cb(0,cb)==0;
}

bool CLogger::m_bEnabledScreen = true;
bool CLogger::m_bEnabledFile = true;
bool CLogger::m_bFileProvided = false;
bool CLogger::m_bInitialized = false;
