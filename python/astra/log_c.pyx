# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://sf.net/projects/astra-toolbox
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------
#
# distutils: language = c++
# distutils: libraries = astra

import six

cdef extern from "astra/Logging.h" namespace "astra":
    cdef enum log_level:
        LOG_DEBUG
        LOG_INFO
        LOG_WARN
        LOG_ERROR

cdef extern from "astra/Logging.h" namespace "astra::CLogger":
    void debug(const char *sfile, int sline, const char *fmt, ...)
    void info(const char *sfile, int sline, const char *fmt, ...)
    void warn(const char *sfile, int sline, const char *fmt, ...)
    void error(const char *sfile, int sline, const char *fmt, ...)
    void setOutputScreen(int fd, log_level m_eLevel)
    void setOutputFile(const char *filename, log_level m_eLevel)
    void enable()
    void enableScreen()
    void enableFile()
    void disable()
    void disableScreen()
    void disableFile()
    void setFormatFile(const char *fmt)
    void setFormatScreen(const char *fmt)

def log_debug(sfile, sline, message):
    cstr = list(map(six.b,(sfile,message)))
    debug(cstr[0],sline,"%s",<char*>cstr[1])

def log_info(sfile, sline, message):
    cstr = list(map(six.b,(sfile,message)))
    info(cstr[0],sline,"%s",<char*>cstr[1])

def log_warn(sfile, sline, message):
    cstr = list(map(six.b,(sfile,message)))
    warn(cstr[0],sline,"%s",<char*>cstr[1])

def log_error(sfile, sline, message):
    cstr = list(map(six.b,(sfile,message)))
    error(cstr[0],sline,"%s",<char*>cstr[1])

def log_enable():
    enable()

def log_enableScreen():
    enableScreen()

def log_enableFile():
    enableFile()

def log_disable():
    disable()

def log_disableScreen():
    disableScreen()

def log_disableFile():
    disableFile()

def log_setFormatFile(fmt):
    cstr = six.b(fmt)
    setFormatFile(cstr)

def log_setFormatScreen(fmt):
    cstr = six.b(fmt)
    setFormatScreen(cstr)

enumList = [LOG_DEBUG,LOG_INFO,LOG_WARN,LOG_ERROR]

def log_setOutputScreen(fd, level):
    setOutputScreen(fd, enumList[level])

def log_setOutputFile(filename, level):
    cstr = six.b(filename)
    setOutputFile(cstr, enumList[level])
