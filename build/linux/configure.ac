dnl -----------------------------------------------------------------------
dnl Copyright: 2010-2014, iMinds-Vision Lab, University of Antwerp
dnl                 2014, CWI, Amsterdam
dnl 
dnl Contact: astra@uantwerpen.be
dnl Website: http://sf.net/projects/astra-toolbox
dnl 
dnl This file is part of the ASTRA Toolbox.
dnl 
dnl 
dnl The ASTRA Toolbox is free software: you can redistribute it and/or modify
dnl it under the terms of the GNU General Public License as published by
dnl the Free Software Foundation, either version 3 of the License, or
dnl (at your option) any later version.
dnl 
dnl The ASTRA Toolbox is distributed in the hope that it will be useful,
dnl but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
dnl GNU General Public License for more details.
dnl 
dnl You should have received a copy of the GNU General Public License
dnl along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
dnl 
dnl -----------------------------------------------------------------------
dnl $Id$

AC_INIT(astra_toolbox, 1.3.0)
AC_CONFIG_SRCDIR([Makefile.in])
LT_INIT([disable-static])

SAVED_CPPFLAGS="$CPPFLAGS"
SAVED_CXXFLAGS="$CXXFLAGS"
SAVED_LDFLAGS="$LDFLAGS"

AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_PROG_MAKE_SET

AC_LANG([C++])

dnl Use iostream to check if the C++ compiler works
AC_CHECK_HEADER(iostream, , AC_MSG_ERROR([No working c++ compiler found]))


# boost-unit-test-framework

AC_MSG_CHECKING([for boost-unit-test-framework])

ASTRA_CHECK_BOOST_UNIT_TEST_FRAMEWORK(-lboost_unit_test_framework-mt, BOOSTUTF=yes_mt, BOOSTUTF=no)
if test x$BOOSTUTF = xno; then
  ASTRA_CHECK_BOOST_UNIT_TEST_FRAMEWORK(-lboost_unit_test_framework, BOOSTUTF=yes, BOOSTUTF=no)
  if test x$BOOSTTHREAD = xno; then
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([No boost-unit-test-framework library found])
  else
    AC_MSG_RESULT([yes, libboost_unit_test_framework])
    LIBS_BOOSTUTF="-lboost_unit_test_framework"
  fi
else
  AC_MSG_RESULT([yes, libboost_unit_test_framework-mt])
  LIBS_BOOSTUTF="-lboost_unit_test_framework-mt"
fi
# TODO: do something with the result


# nvcc, cuda

AC_ARG_WITH(cuda, [[  --with-cuda=path        path of CUDA SDK (optional)]],,)

NVCC_PATH=$PATH
if test x"$with_cuda" != x; then
  NVCC_PATH="$with_cuda/bin:$NVCC_PATH"
fi
AC_PATH_PROG([NVCC], [nvcc], [no], [$NVCC_PATH])
# TODO: do something with the result

HAVECUDA=no
if test x"$NVCC" != xno; then
  HAVECUDA=yes
  BACKUP_CUDA_LDFLAGS="$LDFLAGS"
  if test x"$with_cuda" != x; then
    LDFLAGS_CUDA="-L$with_cuda/lib"
    CPPFLAGS_CUDA="-I$with_cuda/include"
    LDFLAGS="$LDFLAGS $LDFLAGS_CUDA"
  fi
  AC_CHECK_LIB(cudart,cudaMalloc, ,HAVECUDA=no)
  AC_CHECK_LIB(cufft,cufftPlan1d, ,HAVECUDA=no)

  if test x"$HAVECUDA" = xno; then
    # try lib64 instead of lib

    HAVECUDA=yes
    LDFLAGS="$BACKUP_CUDA_LDFLAGS"

    # prevent cached values from being used
    unset ac_cv_lib_cudart_cudaMalloc
    unset ac_cv_lib_cufft_cufftPlan1d

    LDFLAGS_CUDA="-L$with_cuda/lib64"
    LDFLAGS="$LDFLAGS $LDFLAGS_CUDA"
    AC_CHECK_LIB(cudart,cudaMalloc, ,HAVECUDA=no)
    AC_CHECK_LIB(cufft,cufftPlan1d, ,HAVECUDA=no)
  fi

  LDFLAGS="$BACKUP_CUDA_LDFLAGS"
  unset BACKUP_CUDA_LDFLAGS
  # TODO: check for cuda headers?

  AC_SUBST(NVCC)
fi

NVCCFLAGS=""
AC_MSG_CHECKING([if nvcc works])
if test x"$HAVECUDA" = xyes; then
  ASTRA_CHECK_NVCC(HAVECUDA,NVCCFLAGS)
fi
AC_MSG_RESULT($HAVECUDA)

AC_ARG_WITH(cuda_compute, [[  --with-cuda-compute=archs  comma separated list of CUDA compute models (optional)]],,)
if test x"$HAVECUDA" = xyes; then
  AC_MSG_CHECKING([for nvcc archs])
  dnl 10 11 12 13 20 30 32 35 37 50
  if test x"$with_cuda_compute" = x; then
    with_cuda_compute="10,12,20,30,35,50"
  fi
  ASTRA_FIND_NVCC_ARCHS([$with_cuda_compute],NVCCFLAGS,NVCCARCHS)
  AC_MSG_RESULT([$NVCCARCHS])
fi


AC_SUBST(HAVECUDA)
AC_SUBST(LDFLAGS_CUDA)
AC_SUBST(CPPFLAGS_CUDA)
AC_SUBST(NVCCFLAGS)


# mex, matlab

AC_ARG_WITH(matlab, [[  --with-matlab=path      path of Matlab (optional)]],,)

MEX_PATH=$PATH
HAVEMATLAB=no
if test x"$with_matlab" != x; then
  MEX_PATH="$with_matlab/bin:$MEX_PATH"
  AC_PATH_PROG([MEX], [mex], [no], [$MEX_PATH])
  if test x"$MEX" != xno; then
    HAVEMATLAB=yes
    # TODO: check platform of C compiler is same as mex
    AC_SUBST(MEX)
    MATLAB_ROOT="$with_matlab"
    AC_SUBST(MATLAB_ROOT)

    ASTRA_CHECK_MEX_SUFFIX([mexa64 mexglx mexmaci64 mexmaci],[MEXSUFFIX])
    if test x$MEXSUFFIX = x; then
      AC_MSG_FAILURE([Unable to determine matlab mex suffix])
      HAVEMATLAB=no
    fi
    AC_SUBST(MEXSUFFIX)
  fi
fi

AC_SUBST(HAVEMATLAB)


# TODO:

# Detection of tools:
# libtool (how?)

# options:
# debugging/optimization/profiling flags


AC_SUBST(SAVED_CPPFLAGS)
AC_SUBST(SAVED_CXXFLAGS)
AC_SUBST(SAVED_LDFLAGS)


AC_CONFIG_FILES([Makefile])
AC_OUTPUT
