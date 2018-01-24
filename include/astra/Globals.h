/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

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
*/

#ifndef _INC_ASTRA_GLOBALS
#define _INC_ASTRA_GLOBALS

/*! \mainpage The ASTRA-toolbox 
 *
 * <img src="../images/logo_big.png"/>
 */


//----------------------------------------------------------------------------------------

#ifdef _MSC_VER

// disable warning: 'fopen' was declared deprecated
#pragma warning (disable : 4996)
// disable warning: C++ exception handler used, but unwind semantics are not enables
#pragma warning (disable : 4530)
// disable warning: no suitable definition provided for explicit template instantiation request
#pragma warning (disable : 4661)

#endif

//----------------------------------------------------------------------------------------
// standard includes
#include <cassert>
#include <iostream>
#include <fstream>
#include <math.h>

//----------------------------------------------------------------------------------------
// macro's

#define ASTRA_TOOLBOXVERSION_MAJOR 1
#define ASTRA_TOOLBOXVERSION_MINOR 9
#define ASTRA_TOOLBOXVERSION ((ASTRA_TOOLBOXVERSION_MAJOR)*100 + (ASTRA_TOOLBOXVERSION_MINOR))
#define ASTRA_TOOLBOXVERSION_STRING "1.9.0dev"


#define ASTRA_ASSERT(a) assert(a)

#define ASTRA_CONFIG_CHECK(value, type, msg) if (!(value)) { std::cout << "Configuration Error in " << type << ": " << msg << std::endl; return false; }

#define ASTRA_CONFIG_WARNING(type, msg) { std::cout << "Warning in " << type << ": " << msg << sdt::endl; }


#define ASTRA_DELETE(a) if (a) { delete a; a = NULL; }
#define ASTRA_DELETE_ARRAY(a) if (a) { delete[] a; a = NULL; }

#ifdef _MSC_VER

#ifdef DLL_EXPORTS
#define _AstraExport __declspec(dllexport)
#define EXPIMP_TEMPLATE
#else
#define _AstraExport __declspec(dllimport)
#define EXPIMP_TEMPLATE extern
#endif

#else

#define _AstraExport

#endif


//----------------------------------------------------------------------------------------
// typedefs
namespace astra {
	typedef float float32;
	typedef double float64;
	typedef unsigned short int uint16;
	typedef signed short int sint16;
	typedef unsigned char uchar8;
	typedef signed char schar8;

	typedef int int32;
	typedef short int int16;
}

//----------------------------------------------------------------------------------------
// variables
namespace astra {
	const float32 PI = 3.14159265358979323846264338328f;
	const float32 PI32 = 3.14159265358979323846264338328f;
	const float32 PIdiv2 = PI / 2;
	const float32 PIdiv4 = PI / 4;
	const float32 eps = 1e-6f;
	
	extern _AstraExport bool running_in_matlab;
}

//----------------------------------------------------------------------------------------
// structs
namespace astra {
	/**
	 * Struct for storing pixel weigths
	 **/
	struct SPixelWeight
	{
		int m_iIndex;
		float32 m_fWeight;
	};

	/**
	 * Struct combining some properties of a detector in 1D detector row
	 **/
	struct SDetector2D
	{
		int m_iIndex;
		int m_iAngleIndex;
		int m_iDetectorIndex;
	};
	
	/**
	 * Struct combining some properties of a detector in 2D detector array
	 **/
	struct SDetector3D
	{
		int m_iIndex;
		int m_iAngleIndex;
		int m_iDetectorIndex;
		int m_iSliceIndex;
	};
}

namespace astra {
_AstraExport inline int getVersion() { return ASTRA_TOOLBOXVERSION; }
_AstraExport inline const char* getVersionString() { return ASTRA_TOOLBOXVERSION_STRING; }
_AstraExport bool cudaAvailable();
#ifdef ASTRA_CUDA
_AstraExport inline bool cudaEnabled() { return true; }
#else
_AstraExport inline bool cudaEnabled() { return false; }
#endif
}
//----------------------------------------------------------------------------------------
// portability between MSVC and Linux/gcc

#ifndef _MSC_VER
#define EXPIMP_TEMPLATE

#if !defined(FORCEINLINE) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define FORCEINLINE inline __attribute__((__always_inline__))
#else
#define FORCEINLINE inline
#endif

#else

#define FORCEINLINE __forceinline

#endif

//----------------------------------------------------------------------------------------
// use pthreads on Linux and OSX
#if defined(__linux__) || defined(__MACH__)
#define USE_PTHREADS
#endif


#endif
