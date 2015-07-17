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

/** \file astra_mex_log_c.cpp
 *
 *  \brief Manages astra logging
 */
#include <mex.h>
#include "mexHelpFunctions.h"
#include "mexInitFunctions.h"

#include "astra/Logging.h"

using namespace std;
using namespace astra;
//-----------------------------------------------------------------------------------------
/** astra_mex_log('debug', file, line, message);
 *
 * Log a debug message.
 * file: Originating file name
 * line: Originating line number
 * message: Log message.
 */
void astra_mex_log_debug(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string filename = mexToString(prhs[1]);
	int linenumber = (int)mxGetScalar(prhs[2]);
	string message = mexToString(prhs[3]);
	astra::CLogger::debug(filename.c_str(),linenumber,"%s",message.c_str());
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('info', file, line, message);
 *
 * Log an info message.
 * file: Originating file name
 * line: Originating line number
 * message: Log message.
 */
void astra_mex_log_info(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string filename = mexToString(prhs[1]);
	int linenumber = (int)mxGetScalar(prhs[2]);
	string message = mexToString(prhs[3]);
	astra::CLogger::info(filename.c_str(),linenumber,"%s",message.c_str());
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('warn', file, line, message);
 *
 * Log a warning message.
 * file: Originating file name
 * line: Originating line number
 * message: Log message.
 */
void astra_mex_log_warn(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string filename = mexToString(prhs[1]);
	int linenumber = (int)mxGetScalar(prhs[2]);
	string message = mexToString(prhs[3]);
	astra::CLogger::warn(filename.c_str(),linenumber,"%s",message.c_str());
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('error', file, line, message);
 *
 * Log an error message.
 * file: Originating file name
 * line: Originating line number
 * message: Log message.
 */
void astra_mex_log_error(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string filename = mexToString(prhs[1]);
	int linenumber = (int)mxGetScalar(prhs[2]);
	string message = mexToString(prhs[3]);
	astra::CLogger::error(filename.c_str(),linenumber,"%s",message.c_str());
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('enable', type);
 *
 * Enable logging.
 * type: which output to enable ('all', 'file', 'screen')
 */
void astra_mex_log_enable(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string sType = mexToString(prhs[1]);
	if(sType == "all"){
		astra::CLogger::enable();
	}else if(sType == "file"){
		astra::CLogger::enableFile();
	}else if(sType == "screen"){
		astra::CLogger::enableScreen();
	} else {
		mexErrMsgTxt("Specify which output to enable ('all', 'file', or 'screen')");
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('disable', type);
 *
 * Disable logging.
 * type: which output to disable ('all', 'file', 'screen')
 */
void astra_mex_log_disable(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 2) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string sType = mexToString(prhs[1]);
	if(sType == "all"){
		astra::CLogger::disable();
	}else if(sType == "file"){
		astra::CLogger::disableFile();
	}else if(sType == "screen"){
		astra::CLogger::disableScreen();
	} else {
		mexErrMsgTxt("Specify which output to disable ('all', 'file', or 'screen')");
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('format', type, fmt);
 *
 * Enable logging.
 * type: which output to format ('file', 'screen')
 * fmt: format string
 *      Here are the substitutions you may use:
 *      %f: Source file name generating the log call.
 *      %n: Source line number where the log call was made.
 *      %m: The message text sent to the logger (after printf formatting).
 *      %d: The current date, formatted using the logger's date format.
 *      %t: The current time, formatted using the logger's time format.
 *      %l: The log level (one of "DEBUG", "INFO", "WARN", or "ERROR").
 *      %%: A literal percent sign.
 *      The default format string is "%d %t %f(%n): %l: %m\n".
 */
void astra_mex_log_format(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 3) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string sType = mexToString(prhs[1]);
	string sFormat = mexToString(prhs[2]);
	if (!sFormat.empty())
	{
		char lastChar = *sFormat.rbegin();
		if (lastChar!='\n'){
			sFormat += '\n';
		}
	}else{
		sFormat += '\n';
	}
	if(sType == "file"){
		astra::CLogger::setFormatFile(sFormat.c_str());
	}else if(sType == "screen"){
		astra::CLogger::setFormatScreen(sFormat.c_str());
	} else {
		mexErrMsgTxt("Specify which output to format ('file' or 'screen')");
	}
}

//-----------------------------------------------------------------------------------------
/** astra_mex_log('output', type, output, level);
 *
 * Set output file / output screen.
 * type: which output to set ('file', 'screen')
 * output: which output file / screen to use:
 *         'file': filename
 *         'screen': 'stdout' or 'stderr'
 * level: logging level to use ('debug', 'info', 'warn', or 'error')
 */
void astra_mex_log_output(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs < 4) {
		mexErrMsgTxt("Not enough arguments.  See the help document for a detailed argument list. \n");
		return;
	}
	string sType = mexToString(prhs[1]);
	string sOutput = mexToString(prhs[2]);
	string sLevel = mexToString(prhs[3]);
	log_level eLevel;
	if(sLevel == "debug"){
		eLevel = LOG_DEBUG;
	}else if(sLevel == "info"){
		eLevel = LOG_INFO;
	}else if(sLevel == "warn"){
		eLevel = LOG_WARN;
	}else if(sLevel == "error"){
		eLevel = LOG_ERROR;
	}else{
		mexErrMsgTxt("Specify which log level to use ('debug', 'info', 'warn', or 'error')");
	}
	if(sType == "file"){
		astra::CLogger::setOutputFile(sOutput.c_str(),eLevel);
	}else if(sType == "screen"){
		int fd;
		if(sOutput == "stdout"){
			fd=1;
		}else if(sOutput == "stderr"){
			fd=2;
		}else{
			mexErrMsgTxt("Specify which screen to output to ('stdout' or 'stderr')");
		}
		astra::CLogger::setOutputScreen(fd,eLevel);
	} else {
		mexErrMsgTxt("Specify which output to set ('file' or 'screen')");
	}
}

//-----------------------------------------------------------------------------------------
static void printHelp()
{
	mexPrintf("Please specify a mode of operation.\n");
	mexPrintf("Valid modes: debug, info, warn, error, enable, disable, format, output\n");
}

//-----------------------------------------------------------------------------------------
/**
 * ... = astra_mex_log(mode, ...);
 */
void mexFunction(int nlhs, mxArray* plhs[],
				 int nrhs, const mxArray* prhs[])
{
	// INPUT: Mode
	string sMode = "";
	if (1 <= nrhs) {
		sMode = mexToString(prhs[0]);	
	} else {
		printHelp();
		return;
	}

	initASTRAMex();

	// SWITCH (MODE)
	if (sMode == "debug") {
		astra_mex_log_debug(nlhs, plhs, nrhs, prhs);
    }else if (sMode == "info") {
		astra_mex_log_info(nlhs, plhs, nrhs, prhs);
	}else if (sMode == "warn") {
		astra_mex_log_warn(nlhs, plhs, nrhs, prhs);
	}else if (sMode == "error") {
		astra_mex_log_error(nlhs, plhs, nrhs, prhs);
	}else if (sMode == "enable") {
		astra_mex_log_enable(nlhs, plhs, nrhs, prhs);
	}else if (sMode == "disable") {
		astra_mex_log_disable(nlhs, plhs, nrhs, prhs);
	}else if (sMode == "format") {
		astra_mex_log_format(nlhs, plhs, nrhs, prhs);
	}else if (sMode == "output") {
		astra_mex_log_output(nlhs, plhs, nrhs, prhs);
	} else {
		printHelp();
	}
	return;
}
