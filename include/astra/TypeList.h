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

#ifndef _INC_ASTRA_TYPELIST
#define _INC_ASTRA_TYPELIST

#include "Globals.h"
#include <cstring>

namespace astra {
namespace typelist {

	//-----------------------------------------------------------------------------------------
	// basic types
	/**
	 * Type to serve as tail of typelist
	 **/
	class NullType { };
	struct FalseType { enum { value = false }; };
	struct TrueType { enum { value = true }; };

	//-----------------------------------------------------------------------------------------
	// define typelist
	/**
	 * Typelist definition
	 * \par References
	 * [1] Modern C++ design: generic programming and design patterns applied, Andrei Alexandrescu
	 **/
	template <class T, class U>
	struct TypeList
	{
		typedef T Head;
		typedef U Tail;
	};

	//-----------------------------------------------------------------------------------------
	// linearize typelist
	#define TYPELIST_0 NullType
	#define TYPELIST_1(T1)								TypeList<T1, NullType>
	#define TYPELIST_2(T1,T2)							TypeList<T1, TYPELIST_1(T2) >
	#define TYPELIST_3(T1,T2,T3)						TypeList<T1, TYPELIST_2(T2,T3) >
	#define TYPELIST_4(T1,T2,T3,T4)						TypeList<T1, TYPELIST_3(T2,T3,T4) >
	#define TYPELIST_5(T1,T2,T3,T4,T5)					TypeList<T1, TYPELIST_4(T2,T3,T4,T5) >
	#define TYPELIST_6(T1,T2,T3,T4,T5,T6)				TypeList<T1, TYPELIST_5(T2,T3,T4,T5,T6) >
	#define TYPELIST_7(T1,T2,T3,T4,T5,T6,T7)			TypeList<T1, TYPELIST_6(T2,T3,T4,T5,T6,T7) >
	#define TYPELIST_8(T1,T2,T3,T4,T5,T6,T7,T8)			TypeList<T1, TYPELIST_7(T2,T3,T4,T5,T6,T7,T8) >
	#define TYPELIST_9(T1,T2,T3,T4,T5,T6,T7,T8,T9)		TypeList<T1, TYPELIST_8(T2,T3,T4,T5,T6,T7,T8,T9)>
	#define TYPELIST_10(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10)													\
			TypeList<T1, TYPELIST_9(T2,T3,T4,T5,T6,T7,T8,T9,T10) >
	#define TYPELIST_11(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11)												\
			TypeList<T1, TYPELIST_10(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11) >								
	#define TYPELIST_12(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12)											\
			TypeList<T1, TYPELIST_11(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12) >							
	#define TYPELIST_13(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13)										\
			TypeList<T1, TYPELIST_12(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13) >						
	#define TYPELIST_14(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14)									\
			TypeList<T1, TYPELIST_13(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14) >					
	#define TYPELIST_15(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15)								\
			TypeList<T1, TYPELIST_14(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15) >
	#define TYPELIST_16(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16)							\
			TypeList<T1, TYPELIST_15(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16) >
	#define TYPELIST_17(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17)						\
			TypeList<T1, TYPELIST_16(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17) >
	#define TYPELIST_18(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18)					\
			TypeList<T1, TYPELIST_17(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18) >
	#define TYPELIST_19(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19)				\
			TypeList<T1, TYPELIST_18(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19) >
	#define TYPELIST_20(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20)			\
			TypeList<T1, TYPELIST_19(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20) >
	#define TYPELIST_21(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21)		\
			TypeList<T1, TYPELIST_20(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21) >
	#define TYPELIST_22(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22)		\
			TypeList<T1, TYPELIST_21(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22) >
	#define TYPELIST_23(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23)		\
			TypeList<T1, TYPELIST_22(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23) >

	#define TYPELIST_24(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24)		\
			TypeList<T1, TYPELIST_23(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24) >

	#define TYPELIST_25(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25)		\
			TypeList<T1, TYPELIST_24(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25) >

	#define TYPELIST_26(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26)		\
			TypeList<T1, TYPELIST_25(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26) >

	#define TYPELIST_27(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27)		\
			TypeList<T1, TYPELIST_26(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27) >

	#define TYPELIST_28(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28)		\
			TypeList<T1, TYPELIST_27(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28) >

	#define TYPELIST_29(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29)		\
			TypeList<T1, TYPELIST_28(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29) >

	#define TYPELIST_30(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30)		\
			TypeList<T1, TYPELIST_29(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30) >

	#define TYPELIST_31(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31)		\
			TypeList<T1, TYPELIST_30(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31) >

	#define TYPELIST_32(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32)		\
			TypeList<T1, TYPELIST_31(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32) >

	#define TYPELIST_33(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33)		\
			TypeList<T1, TYPELIST_32(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33) >

	#define TYPELIST_34(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34)		\
			TypeList<T1, TYPELIST_33(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34) >

	#define TYPELIST_35(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35)		\
			TypeList<T1, TYPELIST_34(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35) >

	#define TYPELIST_36(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36)		\
			TypeList<T1, TYPELIST_35(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36) >

	#define TYPELIST_37(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37)		\
			TypeList<T1, TYPELIST_36(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37) >

	#define TYPELIST_38(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37,T38)		\
			TypeList<T1, TYPELIST_37(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37,T38) >

	#define TYPELIST_39(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37,T38,T39)		\
			TypeList<T1, TYPELIST_38(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37,T38,T39) >

	#define TYPELIST_40(T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37,T38,T39,T40)		\
			TypeList<T1, TYPELIST_39(T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37,T38,T39,T40) >


	//-----------------------------------------------------------------------------------------
	// calculate length of a typelist
	template <class TList> struct Length;
	template <> struct Length<NullType>
	{
		enum { value = 0 };
	};
	template <class T, class U>
	struct Length< TypeList<T,U> >
	{
		enum { value = 1 + Length<U>::value };
	};

	//-----------------------------------------------------------------------------------------
	// indexed access
	template <class TList, unsigned int i> struct TypeAt;
	template <class Head, class Tail>
	struct TypeAt<TypeList<Head,Tail> , 0>
	{
		typedef Head Result;
	};
	template <class Head, class Tail, unsigned int i>
	struct TypeAt<TypeList<Head,Tail>, i>
	{
		typedef typename TypeAt<Tail, i-1>::Result Result;
	};


	//-----------------------------------------------------------------------------------------
	// append to typelist
	template <class TList, class T> struct Append;
	template <> 
	struct Append<NullType, NullType> {
		typedef NullType Result;
	};
	template <class T> 
	struct Append<NullType, T> {
		typedef TYPELIST_1(T) Result;
	};
	template <class Head, class Tail> 
	struct Append<NullType, TypeList<Head,Tail> > {
		typedef TypeList<Head,Tail> Result;
	};
	template <class Head, class Tail, class T> 
	struct Append<TypeList<Head,Tail>, T> {
		typedef TypeList<Head, typename Append<Tail, T>::Result> Result;
	};

	//-----------------------------------------------------------------------------------------
	// create a new object
	template <class TList> 
	struct CreateObject { 
		template <class U> 
		 static void find (U& functor) { 
			 if (functor(TList::Head::type)) {
				 functor.res = new typename TList::Head();
			 }
			 CreateObject<typename TList::Tail>::find(functor); 
		 }
	}; 
	template <> 
	struct CreateObject<NullType> { 
		template <class U> 
		static void find(U& functor) {} 
	}; 

	template <typename Base>
	struct functor_find {
		functor_find() { res = NULL; }
		bool operator() (string name) { 
			return strcmp(tofind.c_str(), name.c_str()) == 0;
		} 
		string tofind;
		Base* res;
	};




} // end namespace typelist
} // end namespace astra

#endif
