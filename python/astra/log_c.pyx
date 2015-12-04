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

from mpi4py import MPI    
comm    = MPI.COMM_WORLD  
nProcs  = comm.Get_size() 
procId  = comm.Get_rank()  


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
    if procId == 0 : comm.bcast(501, root = 0)
    enable()

def log_enableScreen():
    if procId == 0 : comm.bcast(502, root = 0)
    enableScreen()

def log_enableFile():
    if procId == 0 : comm.bcast(503, root = 0)
    enableFile()

def log_disable():
    if procId == 0 : comm.bcast(504, root = 0)
    disable()

def log_disableScreen():
    if procId == 0 : comm.bcast(505, root = 0)
    disableScreen()

def log_disableFile():
    if procId == 0 : comm.bcast(506, root = 0)
    disableFile()

def log_setFormatFile(fmt):
    if procId == 0 : comm.bcast(507, root = 0)
    fmt = comm.bcast(fmt, root = 0)
    cstr = six.b(fmt)
    setFormatFile(cstr)

def log_setFormatScreen(fmt):
    if procId == 0 : comm.bcast(508, root = 0)
    fmt = comm.bcast(fmt, root = 0)
    cstr = six.b(fmt)
    setFormatScreen(cstr)

enumList = [LOG_DEBUG,LOG_INFO,LOG_WARN,LOG_ERROR]

def log_setOutputScreen(fd, level):
    if procId == 0 : comm.bcast(509, root = 0)
    fd, level = comm.bcast([fd, level], root = 0)
    setOutputScreen(fd, enumList[level])

def log_setOutputFile(filename, level):
    if procId == 0 : comm.bcast(510, root = 0)
    filename, level = comm.bcast([filename, level], root = 0)
    cstr = six.b(filename)
    setOutputFile(cstr, enumList[level])

