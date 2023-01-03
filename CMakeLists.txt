cmake_minimum_required(VERSION 2.8)
project(MODEL_CHECK)

option(USE_MLU  "Build offline examples with CNRT support" ON)
option(USE_DLL "Build sdk via lib mode" ON)
option(USE_STATIC "Generate sdk via static mode" OFF)
option(USE_STATIC_EASYDK "easydk.a will be used." ON)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(CMAKE_TOOLCHAIN_FILE) 
  message("@@ cross compile enabled, CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
  set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_edge)
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
  # add_definitions(-DDEBUG=True)
else()
  set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)  
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
  add_definitions(-DDEBUG=True)
endif()

if(EXISTS $ENV{NEUWARE_HOME})
  set(NEUWARE_HOME $ENV{NEUWARE_HOME})
else()
  if(CMAKE_TOOLCHAIN_FILE)
    set(NEUWARE_HOME "/usr/local/mlu220edge_neuware_v1.7.0" )
    set(MLU220EDGE3RD "/usr/local/mlu220_1.7.0_3rdparty_static")
  else()
    set(NEUWARE_HOME "/usr/local/neuware")
  endif()
endif()

#libai_core_sdk.so 
if (CMAKE_TOOLCHAIN_FILE)
  set(EASYDK_INCLUDE_FILES "/project/workspace/samples/CNStream220edge/easydk/include")
  if (USE_STATIC_EASYDK)
    message("@@ Using static libeasydk")
    set(EASYDK_LIBS "${MLU220EDGE3RD}/easydk/libeasydk.a" )
    # set(EASYDK_LIBS "${MLU220EDGE3RD}/all/libopencv_easydk.a" )
  else()
    message("@@ Using dynamic libeasydk")
    set(EASYDK_LIBS "/project/workspace/samples/CNStream220edge/build/easydk/libeasydk.so" )
  endif()
else()
  set(EASYDK_INCLUDE_FILES "/project/workspace/samples/CNStream/easydk/include")
  set(EASYDK_LIBS "/project/workspace/samples/CNStream/build/easydk/libeasydk.so" )
endif()

message("-- NEUWARE_HOME=${NEUWARE_HOME}")

if(NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${NEUWARE_HOME}/cmake"
    "${NEUWARE_HOME}/cmake/modules"
    )
else()
  message("-- skip CMAKE_MODULE_PATH")
endif()

# -Wfatal-errors 出现错误的时候停止编译
# -Werror 将警告信息当做错误显示
# -Wall 提示警告信息
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_REENTRANT -Wno-sign-compare") #-Wl,--exclude-libs=ALL
if(CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -ldl -Wno-pointer-arith -Wno-narrowing")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pthread -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -Wfatal-errors -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()
if($ENV{USE_ABI} MATCHES "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
elseif($ENV{USE_ABI} MATCHES "0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
  message("-- Use default ABI settings")
endif()

message("@@ CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")


#include and lib
set(CNRT_INCLUDE_FILES "${NEUWARE_HOME}/include")
list(APPEND CNRT_INCLUDE_FILES "${EASYDK_INCLUDE_FILES}")
set(CNRT_LIB_PATHS "${EASYDK_LIBS}")
list(APPEND CNRT_LIB_PATHS "${NEUWARE_HOME}/lib64/libcnrt.so")
list(APPEND CNRT_LIB_PATHS "${NEUWARE_HOME}/lib64/libcncodec.so")
list(APPEND CNRT_LIB_PATHS "${NEUWARE_HOME}/lib64/libcndrv.so")


if(CMAKE_TOOLCHAIN_FILE)
  link_directories("${NEUWARE_HOME}/lib64")
  list(APPEND CNRT_LIB_PATHS ion jpu)
endif()

# ---[

# ---[OpenCV
if(CMAKE_TOOLCHAIN_FILE)
  set(OpenCV_INCLUDE_DIRS ${MLU220EDGE3RD}/opencv/include)
  list(APPEND 3RDPARTY_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
  set(OpenCV_LIB_PATH ${MLU220EDGE3RD}/opencv/lib)
  set(OpenCV_EXTRA_LIB_PATH "${MLU220EDGE3RD}/opencv/share/OpenCV/3rdparty/lib")
  set(OpenCV_LIBS "")
  # list(APPEND OpenCV_LIBS ${MLU220EDGE3RD}/all/libopencv.a)
  # list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_dnn.a)
  # list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_ml.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_video.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_objdetect.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_shape.a)
  # list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_stitching.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_superres.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_videostab.a)
  # list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_calib3d.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_features2d.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_highgui.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_videoio.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_imgcodecs.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_photo.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_imgproc.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_flann.a)
  list(APPEND OpenCV_LIBS ${OpenCV_LIB_PATH}/libopencv_core.a)
  

  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/liblibprotobuf.a)
  list(APPEND OpenCV_LIBS "-Wl,--whole-archive ${OpenCV_EXTRA_LIB_PATH}/libzlib.a -Wl,--no-whole-archive")
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/liblibjpeg-turbo.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/liblibwebp.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/liblibpng.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/liblibtiff.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/liblibjasper.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/libIlmImf.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/libquirc.a)
  list(APPEND OpenCV_LIBS ${OpenCV_EXTRA_LIB_PATH}/libtegra_hal.a)
  message(STATUS "opencv libraries: ${OpenCV_LIBS}")
else()
  set(OpenCV_FIND_QUIETLY true)
  #find_package(OpenCV REQUIRED) //this will include opencv_ts which result in crash on centos
  find_package(OpenCV PATHS "/usr/local/mlu270_3rdparty/opencv" NO_DEFAULT_PATH OPTIONAL_COMPONENTS imgcodecs)
  find_package(OpenCV PATHS "/usr/local/mlu270_3rdparty/opencv" NO_DEFAULT_PATH OPTIONAL_COMPONENTS videoio)
  set(optional_opencv_libs ${OpenCV_LIBS})
  find_package(OpenCV PATHS "/usr/local/mlu270_3rdparty/opencv" NO_DEFAULT_PATH REQUIRED core imgproc highgui features2d video videostab bgsegm)
  # set(OpenCV_INCLUDE_DIRS /usr/incude)
  message("opencv includes: ${OpenCV_INCLUDE_DIRS}")
  list(APPEND 3RDPARTY_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
  if(imgcodecs_FOUND)
    list(APPEND OpenCV_LIBS optional_opencv_libs)
  endif()
    message(STATUS "opencv libraries: ${OpenCV_LIBS}")
endif()

# ---[ Google-gflags
if(CMAKE_TOOLCHAIN_FILE)
  set(GFLAGS_INCLUDE_DIRS ${MLU220EDGE3RD}/gflags/include)
  set(GFLAGS_LIBRARIES ${MLU220EDGE3RD}/gflags/lib/libgflags.a)
else()
  include("${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake")
endif()
list(APPEND COMMON_INCLUDE_FILES ${GFLAGS_INCLUDE_DIRS})
list(APPEND COMMON_LIB_PATHS ${GFLAGS_LIBRARIES})

# ---[ Google-glog
if(CMAKE_TOOLCHAIN_FILE)
  set(GLOG_INCLUDE_DIRS ${MLU220EDGE3RD}/glog/include)
  set(GLOG_LIBRARIES ${MLU220EDGE3RD}/glog/lib/libglog.a)
else()
  include("${PROJECT_SOURCE_DIR}/cmake/FindGlog.cmake")
endif()
list(APPEND COMMON_INCLUDE_FILES ${GLOG_INCLUDE_DIRS})
list(APPEND COMMON_LIB_PATHS ${GLOG_LIBRARIES})

# ---[ jsoncpp
set(JSON_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/jsoncpp")
file(GLOB JSON_CPP "${PROJECT_SOURCE_DIR}/jsoncpp/*.cpp")

# ---[ third_party source code
set(EIGEN_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/third_party/")

message("common link libs: ${COMMON_LIB_PATHS}")
message("cambricon link libs: ${CNRT_LIB_PATHS}")

set(INNER_COMMON_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/inner_utils")
file(GLOB INNER_COMMON_CPP "${INNER_COMMON_INCLUDE_DIRS}/*.cpp")
# aux_source_directory("{PROJECT_SOURCE_DIR}/inner_utils" INNER_COMMON_CPP)
# set(INNER_COMMON_CPP "${PROJECT_SOURCE_DIR}/inner_utils/inner_basic.cpp")

set(UTIL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/utils")
# aux_source_directory("${PROJECT_SOURCE_DIR}/utils" UTIL_CPP)
file(GLOB_RECURSE UTIL_CPP "${UTIL_INCLUDE_DIRS}/*.cpp")

# bytetrack need third_party/Eigen
set(BYTETRACK_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/utils/trackor")
file(GLOB_RECURSE BYTETRACK_CPP "${BYTETRACK_INCLUDE_DIRS}/bytetrack_no_reid/*.cpp" "${BYTETRACK_INCLUDE_DIRS}/bytetrack_origin/*.cpp" )
set(UJSON_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/utils/json_encoder")
file(GLOB_RECURSE UJSON_CPP "${UJSON_INCLUDE_DIRS}/*.cpp")
set(RULE_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/utils/post_rule")
file(GLOB_RECURSE RULE_CPP "${RULE_INCLUDE_DIRS}/*.cpp")

message( "@@ ${INNER_COMMON_CPP} ${UTIL_CPP}")

## PARAM for different platform
add_definitions(-DUSE_MLU=True)
if(CMAKE_TOOLCHAIN_FILE)#MLU220
  add_definitions(-DMLU220=True)
  # USE_STATIC_MODEL(True) 使用固定模型地址, 硬编码进代码中
  add_definitions(-DUSE_STATIC_MODEL)
else()#MLU270
  # add_definitions(-DVERBOSE=True)
  # add_definitions(-DUSE_STATIC_MODEL)
  # add_definitions(-DSIM_MLU220) #模拟mlu220的帧率处理
endif()

EXECUTE_PROCESS(COMMAND git log -1 --pretty=format:%cD<hash>%h<note>%s
        TIMEOUT 100
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE # 删除字符串尾的换行符
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} # 执行路径
        )
message(STATUS "building from git commit, hash:comment ${GIT_VERSION}")
add_definitions(-DBUILD_VERSION=\"${GIT_VERSION}\")

if (USE_DLL)
  message("@@ DLL MODE ACTIVATED")
  if (USE_STATIC)
    add_library(ai_core STATIC libai_core.cpp ${UTIL_CPP} ${INNER_COMMON_CPP} ${UJSON_CPP} ${JSON_CPP} ${BYTETRACK_CPP} ${RULE_CPP})
  else()
    add_definitions(-DUCLOUD_EXPORT)
    message("visibility activated")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")#-fvisibility=hidden
    add_library(ai_core SHARED libai_core.cpp ${UTIL_CPP} ${INNER_COMMON_CPP} ${UJSON_CPP} ${JSON_CPP} ${BYTETRACK_CPP} ${RULE_CPP})
  endif()
  target_include_directories(ai_core PUBLIC ${COMMON_INCLUDE_FILES} ${OpenCV_INCLUDE_DIRS}
                          ${CNRT_INCLUDE_FILES} ${UJSON_INCLUDE_DIRS} ${JSON_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS}  ${BYTETRACK_INCLUDE_DIRS} ${RULE_INCLUDE_DIRS})
  target_link_libraries(ai_core pthread ${COMMON_LIB_PATHS} ${OpenCV_LIBS} ${CNRT_LIB_PATHS})
  set_target_properties(ai_core PROPERTIES PUBLIC_HEADER "libai_core.hpp" )

  # 打架行为识别
  # add_executable(test_fight test_fight.cpp)
  # target_link_libraries(test_fight pthread ai_core)
  # 行人检测+打架行为识别(级联)
  # add_executable(test_fight_cascade test_fight_cascade.cpp)
  # target_link_libraries(test_fight_cascade pthread ai_core)  
  # 高空抛物
  # add_executable(test_gkpw test_gkpw.cpp)
  # target_link_libraries(test_gkpw pthread ai_core)

  # add_executable(unitest unitest.cpp)
  # target_link_libraries(unitest pthread ai_core)  

  # add_executable(unitest_unet_post unitest_unet_post.cpp)
  # target_link_libraries(unitest_unet_post pthread ai_core)

  # 测试多线程表现
  add_executable(test_thread test_thread.cpp)
  target_link_libraries(test_thread pthread ai_core)

  # add_executable(test_case_img_list test_case_img_list.cpp)
  # target_link_libraries(test_case_img_list pthread ai_core)

  add_executable(test_case_vid_new test_case_vid_new.cpp)
  target_link_libraries(test_case_vid_new pthread ai_core)

  # add_executable(batch_infer batch_inference.cpp)
  # target_link_libraries(batch_infer pthread ai_core)

  add_executable(test_one_new test_one_new.cpp)
  target_link_libraries(test_one_new pthread ai_core)

  add_executable(unitest unitest.cpp)
  target_link_libraries(unitest pthread ai_core)  

  add_executable(get_models get_models.cpp)

  add_executable(unitest_mempool unitest_mempool.cpp)
  
  # message("${CMAKE_CXX_FLAGS}")

else()
  # message("@@ CPP MODE ACTIVATED")
  # add_executable(test_yolo_multithread test_yolo_multithread.cpp face_rec.cpp ${UTIL_CPP} ${INNER_COMMON_CPP})
  # target_include_directories(test_yolo_multithread PRIVATE ${COMMON_INCLUDE_FILES} ${OpenCV_INCLUDE_DIRS} ${CNRT_INCLUDE_FILES} ${UTIL_INCLUDE_DIRS} ${INNER_COMMON_INCLUDE_DIRS})
  # target_link_libraries(test_yolo_multithread pthread ${COMMON_LIB_PATHS} ${OpenCV_LIBS} ${CNRT_LIB_PATHS} )
endif()

message(STATUS "building from git commit, hash:comment ${GIT_VERSION}")
message("=========END=========")


