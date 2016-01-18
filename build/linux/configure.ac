dnl -----------------------------------------------------------------------
dnl Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
dnl            2014-2015, CWI, Amsterdam
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

AC_INIT(astra_toolbox, 1.7.1)
AC_CONFIG_SRCDIR([Makefile.in])
LT_INIT([disable-static])

SAVED_CPPFLAGS="$CPPFLAGS"
SAVED_CXXFLAGS="$CXXFLAGS"
SAVED_NVCCFLAGS="$NVCCFLAGS"
SAVED_LDFLAGS="$LDFLAGS"
SAVED_LIBS="$LIBS"

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
HAVEBOOSTUTF=no
if test x$BOOSTUTF = xno; then
  ASTRA_CHECK_BOOST_UNIT_TEST_FRAMEWORK(-lboost_unit_test_framework, BOOSTUTF=yes, BOOSTUTF=no)
  if test x$BOOSTUTF = xno; then
    AC_MSG_RESULT(no)
  else
    AC_MSG_RESULT([yes, libboost_unit_test_framework])
    LIBS_BOOSTUTF="-lboost_unit_test_framework"
    HAVEBOOSTUTF=yes
  fi
else
  AC_MSG_RESULT([yes, libboost_unit_test_framework-mt])
  LIBS_BOOSTUTF="-lboost_unit_test_framework-mt"
  HAVEBOOSTUTF=yes
fi

AC_SUBST(HAVEBOOSTUTF)
AC_SUBST(LIBS_BOOSTUTF)

BOOSTok=yes
AC_CHECK_HEADER([boost/lexical_cast.hpp],[],[BOOSTok=no],[])
AC_CHECK_HEADER([boost/any.hpp],[],[BOOSTok=no],[])
dnl AC_CHECK_HEADER([boost/thread.hpp],[],[BOOSTok=no],[])
dnl AC_CHECK_HEADER([boost/bind.hpp],[],[BOOSTok=no],[])
AC_CHECK_HEADER([boost/static_assert.hpp],[],[BOOSTok=no],[])
AC_CHECK_HEADER([boost/throw_exception.hpp],[],[BOOSTok=no],[])

if test x$BOOSTok = xno; then
  AC_MSG_ERROR([boost not found])
fi


# MPI, multi node implementation 

AC_ARG_WITH(mpi, [[  --with-mpi=path         include path of MPI headers (optional)]],,)

HAVEMPI=no
MPI_PATH=
if test x"$with_mpi" != x -a x"$with_mpi" != xno; then
	HAVEMPI=yes

	MPI_PATH=""
  	if test x"$with_mpi" != x -a x"$with_mpi" != xyes; then
	    MPI_PATH="$with_mpi"
	fi
fi	

AC_SUBST(HAVEMPI)
AC_SUBST(MPI_PATH)


# nvcc, cuda

AC_ARG_WITH(cuda, [[  --with-cuda=path        path of CUDA SDK (optional)]],,)

if test x"$with_cuda" != xno; then
  NVCC_PATH=$PATH
  if test x"$with_cuda" != x -a x"$with_cuda" != xyes; then
    NVCC_PATH="$with_cuda/bin:$NVCC_PATH"
  fi
  AC_PATH_PROG([NVCC], [nvcc], [no], [$NVCC_PATH])
else
  NVCC=no
fi

HAVECUDA=no
if test x"$NVCC" != xno; then
  HAVECUDA=yes
  BACKUP_CUDA_LDFLAGS="$LDFLAGS"
  if test x"$with_cuda" != x -a x"$with_cuda" != xyes; then
    case $host_cpu in
    x86_64)
      LDFLAGS_CUDA="-L$with_cuda/lib64"
      ;;
    *)
      LDFLAGS_CUDA="-L$with_cuda/lib"
      ;;
    esac
    CPPFLAGS_CUDA="-I$with_cuda/include"
    LDFLAGS="$LDFLAGS $LDFLAGS_CUDA"
  fi
  AC_CHECK_LIB(cudart,cudaMalloc, ,HAVECUDA=no)
  AC_CHECK_LIB(cufft,cufftPlan1d, ,HAVECUDA=no)

  LDFLAGS="$BACKUP_CUDA_LDFLAGS"
  unset BACKUP_CUDA_LDFLAGS
  # TODO: check for cuda headers?

  AC_SUBST(NVCC)
fi

if test x"$HAVECUDA" = xyes; then
  AC_MSG_CHECKING([if nvcc works])
  ASTRA_CHECK_NVCC(HAVECUDA,NVCCFLAGS_EXTRA)
  AC_MSG_RESULT($HAVECUDA)
fi

AC_ARG_WITH(cuda_compute, [[  --with-cuda-compute=archs  comma separated list of CUDA compute models (optional)]],,)
if test x"$HAVECUDA" = xyes; then
  AC_MSG_CHECKING([for nvcc archs])
  dnl 10 11 12 13 20 30 32 35 37 50
  if test x"$with_cuda_compute" = x; then
    with_cuda_compute="10,12,20,30,35,50"
  fi
  ASTRA_FIND_NVCC_ARCHS([$with_cuda_compute],NVCCFLAGS_EXTRA,NVCCARCHS)
  AC_MSG_RESULT([$NVCCARCHS])
fi


AC_SUBST(HAVECUDA)
AC_SUBST(LDFLAGS_CUDA)
AC_SUBST(CPPFLAGS_CUDA)
AC_SUBST(NVCCFLAGS_EXTRA)


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
    # TODO: maybe catch mex warnings
    ASTRA_CHECK_MEX_SUFFIX([mexa64 mexglx mexmaci64 mexmaci],[MEXSUFFIX])
    if test x$MEXSUFFIX = x; then
      AC_MSG_FAILURE([Unable to determine matlab mex suffix])
      HAVEMATLAB=no
    fi
    AC_SUBST(MEXSUFFIX)
  fi
fi

AC_SUBST(HAVEMATLAB)

# python

AC_ARG_WITH(python, [[  --with-python=path      path of Python binary (optional)]],,)

HAVEPYTHON=no
if test x"$with_python" != x -a x"$with_python" != xno; then
  if test x"$with_python" = xyes; then
    PYTHON=python
  else
    PYTHON="$with_python"
  fi
  AC_MSG_CHECKING(for python)
  ASTRA_RUN_LOGOUTPUT(echo 'import sys' | $PYTHON -)
  if test $? -ne 0; then
    AC_MSG_ERROR(Python binary not found)
  fi
  AC_MSG_RESULT([$PYTHON])
  HAVEPYTHON=yes
  AC_SUBST(PYTHON)
  AC_MSG_CHECKING(for numpy module)
  ASTRA_TRY_PYTHON([import numpy],,HAVEPYTHON=no)
  if test x$HAVEPYTHON = xno; then
    AC_MSG_RESULT(no)
    AC_MSG_ERROR(You need the numpy module to use the ASTRA toolbox in Python)
  fi
  AC_MSG_RESULT(yes)
  AC_MSG_CHECKING(for Cython module)
  ASTRA_TRY_PYTHON([
import Cython
from distutils.version import LooseVersion
assert(LooseVersion(Cython.__version__)>=LooseVersion("0.13"))
],,HAVEPYTHON=no)
  if test x$HAVEPYTHON = xno; then
    AC_MSG_RESULT(no)
    AC_MSG_ERROR(You need the Cython module (version >=0.13) to use the ASTRA toolbox in Python)
  fi
  AC_MSG_RESULT(yes)
  AC_MSG_CHECKING(for six module)
  ASTRA_TRY_PYTHON([import six],, HAVEPYTHON=no)
  if test x$HAVEPYTHON = xno; then
    AC_MSG_RESULT(no)
    AC_MSG_ERROR(You need the six module to use the ASTRA toolbox in Python)
  fi
  AC_MSG_RESULT(yes)
	
  #Test incase we use MPI
if test x"$with_mpi" != x -a x"$with_mpi" != xno; then
  if test x"$with_mpi" != xno; then
    AC_MSG_CHECKING(for mpi4py module)
    ASTRA_TRY_PYTHON([import mpi4py],, HAVEPYTHON=no)
    if test x$HAVEPYTHON = xno; then
      AC_MSG_RESULT(no)
      AC_MSG_ERROR(You need the mpi4py module to use the ASTRA toolbox with MPI support in Python)
    fi
    AC_MSG_RESULT(yes)
    AC_MSG_CHECKING(for dill module)
    ASTRA_TRY_PYTHON([import dill],, HAVEPYTHON=no)
    if test x$HAVEPYTHON = xno; then
      AC_MSG_RESULT(no)
      AC_MSG_ERROR(You need the dill module to use the ASTRA toolbox with MPI support in Python)
    fi
    AC_MSG_RESULT(yes)
    AC_MSG_CHECKING(for scipy module)
    ASTRA_TRY_PYTHON([import scipy],, HAVEPYTHON=no)
    if test x$HAVEPYTHON = xno; then
      AC_MSG_RESULT(no)
      AC_MSG_ERROR(You need the scipy module to use the ASTRA toolbox with MPI support in Python)
    fi
    AC_MSG_RESULT(yes)
  fi
fi
#end MPI tests

fi

AC_SUBST(HAVEPYTHON)

#OS specific setup
AC_CANONICAL_HOST
case $host_os in
  darwin* )
        CPPFLAGS_OS="-stdlib=libstdc++ -mmacosx-version-min=10.6"
        ;;
    *)
        CPPFLAGS_OS=""
        ;;
esac

AC_SUBST(CPPFLAGS_OS)


# TODO:

# Detection of tools:
# libtool (how?)

# options:
# debugging/optimization/profiling flags


AC_SUBST(SAVED_CPPFLAGS)
AC_SUBST(SAVED_CXXFLAGS)
AC_SUBST(SAVED_NVCCFLAGS)
AC_SUBST(SAVED_LDFLAGS)
AC_SUBST(SAVED_LIBS)
AC_CONFIG_FILES([Makefile])
AC_OUTPUT

echo
echo "Summary of ASTRA Toolbox build options:"
echo "  CUDA   : $HAVECUDA"
echo "  Matlab : $HAVEMATLAB"
echo "  Python : $HAVEPYTHON"
echo "  MPI    : $HAVEMPI"
echo
echo "  prefix : $prefix"
echo
