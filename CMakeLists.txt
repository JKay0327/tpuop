cmake_minimum_required(VERSION 3.10)
project(tpuop VERSION 3.1.0)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-omit-frame-pointer -fsanitize=address -fno-omit-frame-pointer ")

set(LIBSOPHON /opt/sophon/libsophon-current)
include_directories(${LIBSOPHON}/include)
link_directories(${LIBSOPHON}/lib)
message(STATUS "USE_LIBSOPHON: ${USE_LIBSOPHON}")

include_directories(src)
add_library(tpuop SHARED src/tpuop.cc)
target_link_libraries(tpuop )
target_link_libraries(tpuop bmlib bmrt libbmodel.a pthread )