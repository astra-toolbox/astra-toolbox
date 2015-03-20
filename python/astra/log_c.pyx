#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pyastratoolbox/
#
#
#This file is part of the Python interface to the
#All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
#The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
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
    debug(six.b(sfile),sline,six.b(message))

def log_info(sfile, sline, message):
    info(six.b(sfile),sline,six.b(message))

def log_warn(sfile, sline, message):
    warn(six.b(sfile),sline,six.b(message))

def log_error(sfile, sline, message):
    error(six.b(sfile),sline,six.b(message))

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
    setFormatFile(six.b(fmt))

def log_setFormatScreen(fmt):
    setFormatScreen(six.b(fmt))

enumList = [LOG_DEBUG,LOG_INFO,LOG_WARN,LOG_ERROR]

def log_setOutputScreen(fd, level):
    setOutputScreen(fd, enumList[level])

def log_setOutputFile(filename, level):
    setOutputFile(six.b(filename), enumList[level])