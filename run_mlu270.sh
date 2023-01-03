#!/bin/bash
rm ./build/*.so
rm ./build/*.a
cd ./build
rm -r CMake*
rm -r cmake*

# echo 'generate static'
# cmake -DUSE_STATIC_EASYDK=ON -DUSE_STATIC=ON -DUSE_DLL=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/cross-compile.cmake ..
# make -j24

# rm -r CMake*
# rm -r cmake*

# echo 'MLU270::generate dynamic'
# cmake -DUSE_STATIC=OFF -DUSE_DLL=ON ..
# make -j42




