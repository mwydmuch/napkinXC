# napkinXML
# Authors: Marek Wydmuch

cmake_minimum_required(VERSION 3.1.0)
project(napkinXML)

# Project version
set(napkinXML_VERSION_MAJOR 0)
set(napkinXML_VERSION_MINOR 3)
set(napkinXML_PATCH_VERSION 0)

# Building options
option(WITH_MIPS_EXT "Build extension with Maximum Inner Product Search-based models" OFF)

# Configure file
configure_file(
    ${CMAKE_SOURCE_DIR}/src/version.in
    ${CMAKE_SOURCE_DIR}/src/version.h
)

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

file(GLOB SOURCES
    src/*.cpp
    src/blas/*.c
    src/data_readers/*.cpp
    src/liblinear/*.cpp
    src/models/*.cpp)

add_executable(nxml ${SOURCES})

target_include_directories(nxml PUBLIC
    src
    src/blas
    src/data_readers
    src/liblinear
    src/mips_models
    src/models)

if (WITH_MIPS_EXT)
    file(GLOB MIPS_EXT_SOURCES src/mips_models/*.cpp)
    target_sources(nxml PUBLIC ${MIPS_EXT_SOURCES})
    link_directories(${CMAKE_SOURCE_DIR}/nmslib/similarity_search/release)
    target_include_directories(nxml PUBLIC nmslib/similarity_search/include)
    target_link_libraries(nxml PUBLIC NonMetricSpaceLib)
endif ()

set_target_properties(nxml
    PROPERTIES
    OUTPUT_NAME nxml
    PROJECT_LABEL "napkinXML")
