cmake_minimum_required(VERSION 3.30)
project(simd_bvh)

set(CMAKE_CXX_STANDARD 20)

option(ENABLE_PERF_FLAGS "Enable performance flags" OFF)
option(RUN_TESTS "Run tests" ON)

if (ENABLE_PERF_FLAGS)
    if (MSVC)
        message(STATUS "[JTX] Using MSVC compiler")
        add_compile_options(/O2 /fp:fast)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "[JTX] Using GNU or Clang compiler")
        add_compile_options(-O3 -ffast-math)
    else ()
        message(WARNING "[JTX] Unknown compiler. No optimization flags added.")
    endif ()
endif ()

if (RUN_TESTS)
    add_compile_definitions(RUN_TESTS)
endif ()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    message(STATUS "[JTX] Targeting ARM architecture - enabling NEON")
    add_compile_definitions(USE_NEON)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i386|i686")
    message(STATUS "[JTX] Targeting x86 architecture - enabling SSE")
    add_compile_definitions(USE_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2") # Enable SSE4.2 for x86
endif()

add_subdirectory(ext/jtxlib)

set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(ext/assimp)

add_executable(simd_bvh src/main.cpp
        src/aabb.hpp
        src/scene.hpp
        src/common.hpp
        src/mesh.hpp
        src/scene.cpp
        src/bvh2.hpp
        src/bvh2.cpp
        src/primitives.hpp
        src/bvh4.hpp
        src/simd.hpp
        src/tests.hpp
        src/tests.cpp
        src/bvh4.cpp
)

target_link_libraries(simd_bvh PRIVATE jtxlib assimp)

target_include_directories(simd_bvh
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ext/jtxlib/src
)

if (WIN32)
    add_custom_command(TARGET simd_bvh POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:assimp>
            $<TARGET_FILE_DIR:jtxlib>
    )
endif ()