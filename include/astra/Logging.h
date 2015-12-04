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

#ifndef _INC_ASTRA_LOGGING
#define _INC_ASTRA_LOGGING

#include "astra/Globals.h"

#define ASTRA_DEBUG(...) astra::CLogger::debug(__FILE__,__LINE__, __VA_ARGS__)
#define ASTRA_INFO(...) astra::CLogger::info(__FILE__,__LINE__, __VA_ARGS__)
#define ASTRA_WARN(...) astra::CLogger::warn(__FILE__,__LINE__, __VA_ARGS__)
#define ASTRA_ERROR(...) astra::CLogger::error(__FILE__,__LINE__, __VA_ARGS__)

namespace astra
{

enum log_level {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
};

class _AstraExport CLogger
{
	CLogger();
  ~CLogger();
  static bool m_bEnabledFile;
  static bool m_bEnabledScreen;
  static bool m_bFileProvided;
  static bool m_bInitialized;
  static void _assureIsInitialized();
  static void _setLevel(int id, log_level m_eLevel);

public:

	/**
	 * Writes a line to the log file (newline is added). Ignored if logging is turned off.
	 *
	 * @param sfile
   * The name of the source file making this log call (e.g. __FILE__).
   *
   * @param sline
   * The line number of the call in the source code (e.g. __LINE__).
   *
   * @param id
   * The id of the logger to write to.
   *
   * @param fmt
   * The format string for the message (printf formatting).
   *
   * @param ...
   * Any additional format arguments.
	 */
  static void debug(const char *sfile, int sline, const char *fmt, ...);
  static void info(const char *sfile, int sline, const char *fmt, ...);
  static void warn(const char *sfile, int sline, const char *fmt, ...);
  static void error(const char *sfile, int sline, const char *fmt, ...);

  /**
	 * Sets the file to log to, with logging level.
   *
   * @param filename
   * File to log to.
	 *
	 * @param m_eLevel
   * Logging level (LOG_DEBUG, LOG_WARN, LOG_INFO, LOG_ERROR).
   *
	 */
  static void setOutputFile(const char *filename, log_level m_eLevel);

  /**
	 * Sets the screen to log to, with logging level.
   *
   * @param screen_fd
   * Screen file descriptor (1 for stdout, 2 for stderr)
	 *
	 * @param m_eLevel
   * Logging level (LOG_DEBUG, LOG_WARN, LOG_INFO, LOG_ERROR).
   *
	 */
  static void setOutputScreen(int fd, log_level m_eLevel);
  
  /**
   * Set the format string for log messages.  Here are the substitutions you may
   * use:
   *
   *     %f: Source file name generating the log call.
   *     %n: Source line number where the log call was made.
   *     %m: The message text sent to the logger (after printf formatting).
   *     %d: The current date, formatted using the logger's date format.
   *     %t: The current time, formatted using the logger's time format.
   *     %l: The log level (one of "DEBUG", "INFO", "WARN", or "ERROR").
   *     %D: The MPI runtime information.
   *     %%: A literal percent sign.
   *
   * The default format string is "%d %t %f(%n): %l: %m\n".
   *
   * IF and only IF we run with more than one process the default is
   * The default format string is "%D %d %t %f(%n): %l: %m\n".
   *
   * @param fmt
   * The new format string, which must be less than 256 bytes.
   * You probably will want to end this with a newline (\n).
   *
   */
  static void setFormatFile(const char *fmt);
  static void setFormatScreen(const char *fmt);


  /**
   * Enable logging.
   *
   */
  static void enable();
  static void enableScreen();
  static void enableFile();

  /**
   * Disable logging.
   *
   */
  static void disable();
  static void disableScreen();
  static void disableFile();

  /**
   * Set callback function for logging to screen.
   * @return whether callback was set succesfully.
   *
   */
  static bool setCallbackScreen(void (*cb)(const char *msg, size_t len));

};

}

#endif /* _INC_ASTRA_LOGGING */
