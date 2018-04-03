cmake_minimum_required(VERSION 2.8.9)
project(napkinXML)

set (fasttext_VERSION_MAJOR 0)
set (fasttext_VERSION_MINOR 1)

include_directories(napkinXML)

set(CMAKE_CXX_FLAGS " -std=c++11 -pthread -funroll-loops -O3")

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
