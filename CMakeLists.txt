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

# Configure file
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

configure_file(
    ${SRC_DIR}/version.in
    ${SRC_DIR}/version.h
)
# Gather napkinXML source files
file(GLOB SOURCES
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/blas/*.c
    ${SRC_DIR}/data_readers/*.cpp
    ${SRC_DIR}/liblinear/*.cpp
    ${SRC_DIR}/models/*.cpp)

set(INCLUDES
    ${SRC_DIR}
    ${SRC_DIR}/blas
    ${SRC_DIR}/data_readers
    ${SRC_DIR}/liblinear
    ${SRC_DIR}/models)

set(LIBRARIES)

# MIPS extension files
if (WITH_MIPS_EXT)
    link_directories(${CMAKE_SOURCE_DIR}/nmslib/similarity_search/release)

    file(GLOB MIPS_EXT_SOURCES ${SRC_DIR}/mips_models/*.cpp)
    set(MIPS_EXT_INCLUDES
        ${SRC_DIR}/mips_models
        ${CMAKE_SOURCE_DIR}/nmslib/similarity_search/include)

    list(APPEND SOURCES ${MIPS_EXT_SOURCES})
    list(APPEND INCLUDES ${MIPS_EXT_INCLUDES})
    list(APPEND LIBRARIES NonMetricSpaceLib)
endif ()

add_executable(nxml ${SOURCES})
target_include_directories(nxml PUBLIC ${INCLUDES})
target_link_libraries(nxml PUBLIC ${LIBRARIES})

set_target_properties(nxml
    PROPERTIES
    OUTPUT_NAME nxml
    PROJECT_LABEL "napkinXML")
