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
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>

//----------------------------------------------------------------------------------------
// macro's

#define ASTRA_TOOLBOXVERSION_MAJOR 1
#define ASTRA_TOOLBOXVERSION_MINOR 7
#define ASTRA_TOOLBOXVERSION ((ASTRA_TOOLBOXVERSION_MAJOR)*100 + (ASTRA_TOOLBOXVERSION_MINOR))
#define ASTRA_TOOLBOXVERSION_STRING "1.7.1"


#define ASTRA_ASSERT(a) assert(a)

#define ASTRA_CONFIG_CHECK(value, type, msg) if (!(value)) { cout << "Configuration Error in " << type << ": " << msg << endl; return false; }

#define ASTRA_CONFIG_WARNING(type, msg) { cout << "Warning in " << type << ": " << msg << endl; }


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
// globals vars & functions
//namespace astra {
//#define ToolboxVersion 0.1f;

//float32 getVersion() { return ToolboxVersion; }

//_AstraExport bool cudaEnabled() { 
//#ifdef ASTRA_CUDA	
//	return true; 
//#else
//	return false;
//#endif
//}
//}

//----------------------------------------------------------------------------------------
// errors
namespace astra {

typedef enum {ASTRA_SUCCESS, 
			  ASTRA_ERROR_NOT_INITIALIZED,
			  ASTRA_ERROR_INVALID_FILE,
			  ASTRA_ERROR_OUT_OF_RANGE,
			  ASTRA_ERROR_DIMENSION_MISMATCH,
			  ASTRA_ERROR_EXTERNAL_LIBRARY,
			  ASTRA_ERROR_ALLOCATION,
			  ASTRA_ERROR_NOT_IMPLEMENTED} AstraError;
}


//----------------------------------------------------------------------------------------
// variables
namespace astra {
	const float32 PI = 3.14159265358979323846264338328f;
	const float32 PI32 = 3.14159265358979323846264338328f;
	const float32 PIdiv2 = PI / 2;
	const float32 PIdiv4 = PI / 4;
	const float32 eps = 1e-7f;
	
	extern _AstraExport bool running_in_matlab;
}

//----------------------------------------------------------------------------------------
// math
namespace astra {

	inline float32 cos_73s(float32 x) 
	{ 
		/*
		const float32 c1 =  0.999999953464f;
		const float32 c2 = -0.4999999053455f;
		const float32 c3 =  0.0416635846769f;
		const float32 c4 = -0.0013853704264f;
		const float32 c5 =  0.000023233f;
		*/
		const float c1= (float)0.99940307;
		const float c2= (float)-0.49558072;
		const float c3= (float)0.03679168;

		float32 x2;
		x2 = x * x;
		//return (c1 + x2*(c2 + x2*(c3 + x2*(c4 + c5*x2))));
		return (c1 + x2*(c2 + c3 * x2));
	}

	inline float32 fast_cos(float32 x) 
	{
		int quad; 

		//x = fmod(x, 2*PI);		// Get rid of values > 2* pi
		if (x < 0) x = -x;		// cos(-x) = cos(x)
		quad = int(x/PIdiv2);	// Get quadrant # (0 to 3) 
		switch (quad) {
			case 0: return  cos_73s(x);
			case 1: return -cos_73s(PI-x);
			case 2: return -cos_73s(x-PI);
			case 3: return  cos_73s(2*PI-x);
		}
		return 0.0f;
	}

	inline float32 fast_sin(float32 x){
		return fast_cos(PIdiv2-x);
	}

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
//----------------------------------------------------------------------------------------
// some toys

// safe reinterpret cast
template <class To, class From>
To safe_reinterpret_cast(From from)
{
	BOOST_STATIC_ASSERT(sizeof(From) <= sizeof(To));
	return reinterpret_cast<To>(from);
}

//----------------------------------------------------------------------------------------
// functions for testing
template<typename T>
inline void writeArray(T*** arr, int dim1, int dim2, int dim3, const std::string& filename)
{
	std::ofstream out(filename.c_str());
	int i1, i2, i3;
	for (i1 = 0; i1 < dim1; ++i1) {
		for (i2 = 0; i2 < dim2; ++i2) {
			for (i3 = 0; i3 < dim3; ++i3) {
				out << arr[i1][i2][i3] << " ";
			}
			out << std::endl;
		}
		out << std::endl;
	}
	out.close();
}

template<typename T>
inline void writeArray(T** arr, int dim1, int dim2, const std::string& filename)
{
	std::ofstream out(filename.c_str());
	for (int i1 = 0; i1 < dim1; i1++) {
		for (int i2 = 0; i2 < dim2; i2++) {
			out << arr[i1][i2] << " ";
		}
		out << std::endl;
	}
	out.close();
}

template<typename T>
inline void writeArray(T* arr, int dim1, const std::string& filename)
{
	std::ofstream out(filename.c_str());
	for (int i1 = 0; i1 < dim1; i1++) {
		out << arr[i1] << " ";
	}
	out.close();
}
namespace astra {
_AstraExport inline int getVersion() { return ASTRA_TOOLBOXVERSION; }
_AstraExport inline const char* getVersionString() { return ASTRA_TOOLBOXVERSION_STRING; }
#ifdef ASTRA_CUDA
_AstraExport inline bool cudaEnabled() { return true; }
#else
_AstraExport inline bool cudaEnabled() { return false; }
#endif
}
//----------------------------------------------------------------------------------------
// portability between MSVC and Linux/gcc

#ifndef _MSC_VER
#include "swrap.h"
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
