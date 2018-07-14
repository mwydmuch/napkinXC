cmake_minimum_required(VERSION 2.8.9)
project(napkinXML)

set(napkinXML_VERSION_MAJOR 0)
set(napkinXML_VERSION_MINOR 1)
set(napkinXML_PATCH_VERSION 0)

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

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/liblinear/*.cpp
    src/blas/*.c)

add_executable(nxml ${SOURCES})

target_include_directories(nxml PUBLIC
    src
    src/liblinear
    src/blas)

set_target_properties(nxml
    PROPERTIES
    OUTPUT_NAME nxml
    PROJECT_LABEL "napkinXML")
