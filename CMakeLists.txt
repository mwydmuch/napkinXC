# napkinXML
# Authors: Marek Wydmuch

cmake_minimum_required(VERSION 2.8.9)
project(napkinXML)

# Project version
set(napkinXML_VERSION_MAJOR 0)
set(napkinXML_VERSION_MINOR 1)
set(napkinXML_PATCH_VERSION 0)
set(napkinXML_VERSION ${napkinXML_VERSION_MAJOR}.${napkinXML_VERSION_MINOR}.${napkinXML_PATCH_VERSION})

# Building options
option(BUILD_MIPS_EXT "Build with maximum inner product search based models" OFF)

set(CMAKE_CXX_STANDARD 11)

# Set the release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add pthread for Linux
if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Add more aggressive optimization for Release build
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -O3")
endif(CMAKE_BUILD_TYPE MATCHES Release)

#include_directories(${CMAKE_SOURCE_DIR}/nmslib/similarity_search/include)
link_directories(${CMAKE_SOURCE_DIR}/nmslib/similarity_search/release)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/blas/*.c
    src/data_readers/*.cpp
    src/liblinear/*.cpp
    src/mips_models/*.cpp
    src/models/*.cpp
    )

add_executable(nxml ${SOURCES})

target_include_directories(nxml PUBLIC
    src
    src/blas
    src/data_readers
    src/liblinear
    src/mips_models
    src/models
    nmslib/similarity_search/include
    )

target_link_libraries(nxml PUBLIC
        NonMetricSpaceLib
        #gsl
        #gslcblas
        #lshkit
    )

set_target_properties(nxml
    PROPERTIES
    OUTPUT_NAME nxml
    PROJECT_LABEL "napkinXML")
