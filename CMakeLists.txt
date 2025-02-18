cmake_minimum_required(VERSION 3.30)
project(simd_bvh)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(ext/jtxlib)

add_executable(simd_bvh main.cpp)

target_link_libraries(simd_bvh PRIVATE jtxlib)

target_include_directories(simd_bvh
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ext/jtxlib/src
)