AC_DEFUN([ASTRA_CHECK_BOOST_THREAD],[
BOOST_BACKUP_LIBS="$LIBS"
LIBS="$LIBS $1"
AC_LINK_IFELSE([AC_LANG_SOURCE([
#include <boost/thread.hpp>
int main()
{
  boost::thread t;
  boost::posix_time::milliseconds m(1);
  t.timed_join(m);
  return 0;
}
])],[$2],[$3])
LIBS="$BOOST_BACKUP_LIBS"
unset BOOST_BACKUP_LIBS
])

AC_DEFUN([ASTRA_CHECK_BOOST_UNIT_TEST_FRAMEWORK],[
BOOST_BACKUP_LIBS="$LIBS"
LIBS="$LIBS $1"
AC_LINK_IFELSE([AC_LANG_SOURCE([
#define BOOST_TEST_DYN_LINK

#define BOOST_AUTO_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
])],[$2],[$3])
LIBS="$BOOST_BACKUP_LIBS"
unset BOOST_BACKUP_LIBS
])

dnl ASTRA_CHECK_MEX_SUFFIX(list-of-suffices, variable-to-set)
AC_DEFUN([ASTRA_CHECK_MEX_SUFFIX],[
cat >conftest.cc <<_ACEOF
extern "C" void mexFunction() {
}
_ACEOF
$CXX -fPIC -c -o conftest.o conftest.cc
$MEX -cxx -output conftest conftest.o
$2=""
for suffix in $1; do
  if test -f "conftest.$suffix"; then
    $2="$suffix"
    rm -f "conftest.$suffix"
  fi
done
rm -f conftest.cc conftest.o
])

dnl ASTRA_CHECK_NVCC(variable-to-set,cppflags-to-set)
AC_DEFUN([ASTRA_CHECK_NVCC],[
cat >conftest.cu <<_ACEOF
#include <iostream>
int main() {
  std::cout << "Test" << std::endl;
  return 0;
}
_ACEOF
$1="yes"
$NVCC -c -o conftest.o conftest.cu $$2 >conftest.nvcc.out 2>&1 || {
  $1="no"
  # Check if hack for gcc 4.4 helps
  if grep -q __builtin_stdarg_start conftest.nvcc.out; then
    NVCC_OPT="-Xcompiler -D__builtin_stdarg_start=__builtin_va_start"
    $NVCC -c -o conftest.o conftest.cu $$2 $NVCC_OPT >/dev/null 2>&1 && {
      $1="yes"
      $2="$$2 $NVCC_OPT"
    }
  fi
}
rm -f conftest.cu conftest.o conftest.nvcc.out
])
