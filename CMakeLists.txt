cmake_minimum_required(VERSION 2.8.9)
project(napkinXML)

set(napkinXML_VERSION_MAJOR 0)
set(napkinXML_VERSION_MINOR 1)
set(napkinXML_PATCH_VERSION 0)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS " -pthread -funroll-loops -O3")

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/liblinear/*.cpp
    src/blas/*.c)

add_executable(nxml ${SOURCES} src/online_training.h src/online_training.cpp)

target_include_directories(nxml PUBLIC
    src
    src/liblinear
    src/blas)

set_target_properties(nxml
    PROPERTIES
    OUTPUT_NAME nxml
    PROJECT_LABEL "napkinXML")
