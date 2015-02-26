#!/bin/bash

aclocal
if test $? -ne 0; then
  echo "Error running aclocal"
  exit 1
fi

autoconf
if test $? -ne 0; then
  echo "Error running autoconf"
  exit 1
fi

case `uname` in Darwin*) LIBTOOLIZEBIN=glibtoolize ;;
  *) LIBTOOLIZEBIN=libtoolize ;; esac

$LIBTOOLIZEBIN --install --force > /dev/null 2>&1
if test $? -ne 0; then
  $LIBTOOLIZEBIN --force
  if test $? -ne 0; then
    echo "Error running libtoolize"
    exit 1
  fi
fi

if test ! -e config.guess; then
  ln -s config.guess.dist config.guess
fi

if test ! -e config.sub; then
  ln -s config.sub.dist config.sub
fi

if test ! -e install-sh; then
  ln -s install-sh.dist install-sh
fi

echo "Done."
