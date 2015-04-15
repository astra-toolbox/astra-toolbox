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

from . import log_c as l

import inspect

def debug(message):
    """Log a debug message.
    
    :param message: Message to log.
    :type message: :class:`string`
    """
    prev_f = inspect.getframeinfo(inspect.currentframe().f_back)
    l.log_debug(prev_f.filename,prev_f.lineno,message)

def info(message):
    """Log an info message.
    
    :param message: Message to log.
    :type message: :class:`string`
    """
    prev_f = inspect.getframeinfo(inspect.currentframe().f_back)
    l.log_info(prev_f.filename,prev_f.lineno,message)

def warn(message):
    """Log a warning message.
    
    :param message: Message to log.
    :type message: :class:`string`
    """
    prev_f = inspect.getframeinfo(inspect.currentframe().f_back)
    l.log_warn(prev_f.filename,prev_f.lineno,message)

def error(message):
    """Log an error message.
    
    :param message: Message to log.
    :type message: :class:`string`
    """
    prev_f = inspect.getframeinfo(inspect.currentframe().f_back)
    l.log_error(prev_f.filename,prev_f.lineno,message)

def enable():
    """Enable logging to screen and file."""
    l.log_enable()

def enableScreen():
    """Enable logging to screen."""
    l.log_enableScreen()

def enableFile():
    """Enable logging to file (note that a file has to be set)."""
    l.log_enableFile()

def disable():
    """Disable all logging."""
    l.log_disable()

def disableScreen():
    """Disable logging to screen."""
    l.log_disableScreen()

def disableFile():
    """Disable logging to file."""
    l.log_disableFile()

def setFormatFile(fmt):
    """Set the format string for log messages.  Here are the substitutions you may use:
    
    %f: Source file name generating the log call.
    %n: Source line number where the log call was made.
    %m: The message text sent to the logger (after printf formatting).
    %d: The current date, formatted using the logger's date format.
    %t: The current time, formatted using the logger's time format.
    %l: The log level (one of "DEBUG", "INFO", "WARN", or "ERROR").
    %%: A literal percent sign.
    
    The default format string is "%d %t %f(%n): %l: %m\n".
    
    :param fmt: Format to use, end with "\n".
    :type fmt: :class:`string`
    """
    l.log_setFormatFile(fmt)

def setFormatScreen(fmt):
    """Set the format string for log messages.  Here are the substitutions you may use:
    
    %f: Source file name generating the log call.
    %n: Source line number where the log call was made.
    %m: The message text sent to the logger (after printf formatting).
    %d: The current date, formatted using the logger's date format.
    %t: The current time, formatted using the logger's time format.
    %l: The log level (one of "DEBUG", "INFO", "WARN", or "ERROR").
    %%: A literal percent sign.
    
    The default format string is "%d %t %f(%n): %l: %m\n".
    
    :param fmt: Format to use, end with "\n".
    :type fmt: :class:`string`
    """
    l.log_setFormatScreen(fmt)

STDOUT=1
STDERR=2

DEBUG=0
INFO=1
WARN=2
ERROR=3

def setOutputScreen(fd, level):
    """Set which screen to output to, and which level to use.
    
    :param fd: File descriptor of output screen (STDOUT or STDERR).
    :type fd: :class:`int`
    :param level: Logging level to use (DEBUG, INFO, WARN, or ERROR).
    :type level: :class:`int`
    """
    l.log_setOutputScreen(fd, level)

def setOutputFile(filename, level):
    """Set which file to output to, and which level to use.
    
    :param filename: File name of output file.
    :type filename: :class:`string`
    :param level: Logging level to use (DEBUG, INFO, WARN, or ERROR).
    :type level: :class:`int`
    """
    l.log_setOutputFile(filename, level)