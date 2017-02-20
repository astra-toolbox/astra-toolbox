#!/bin/sh

cd $SRC_DIR/python/
CPPFLAGS="-DASTRA_CUDA -DASTRA_PYTHON $CPPFLAGS -I$SRC_DIR/ -I$SRC_DIR/include -I$CUDA_ROOT/include" CC=$CC python ./builder.py build install
