#!/bin/bash
rm ./build_edge/*.so
rm ./build_edge/*.a
cd ./build_edge
rm -r CMake*
rm -r cmake*

# echo 'generate static'
# cmake -DUSE_STATIC_EASYDK=ON -DUSE_STATIC=ON -DUSE_DLL=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/cross-compile.cmake ..
# make -j24

# rm -r CMake*
# rm -r cmake*

echo 'generate dynamic'
cmake -DUSE_STATIC_EASYDK=ON -DUSE_STATIC=OFF -DUSE_DLL=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/cross-compile.cmake ..
make -j42

cp libai_core.so /project/workspace/samples/face_rec_sdk_demo/lib/libai_core/lib
cp libai_core.so /project/workspace/samples/face_rec_sdk_demo/build_edge
cp ../libai_core.hpp /project/workspace/samples/face_rec_sdk_demo/lib/libai_core/include
# cp ../libai_core_common.hpp /project/workspace/samples/face_rec_sdk_demo/lib/libai_core/include



