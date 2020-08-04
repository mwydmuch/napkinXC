# napkinXC
# Authors: Marek Wydmuch

cmake_minimum_required(VERSION 3.12)
project(napkinXC
    VERSION 0.4.0
    DESCRIPTION "Extremely simple and fast extreme multi-class and multi-label classifiers"
    HOMEPAGE_URL https://github.com/mwydmuch/napkinXC
    LANGUAGES C CXX)

# Building options
option(EXE "Build executable" ON)
option(PYTHON "Build Python binding" OFF)
option(WITH_MIPS_EXT "Build extension with Maximum Inner Product Search-based models" OFF)

set(CMAKE_CXX_STANDARD 17)

# Set the release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Add more aggressive optimization for Release build TODO: improve this
if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -O3")
endif(CMAKE_BUILD_TYPE MATCHES Release)

# Configure file
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

configure_file(
    ${SRC_DIR}/version.in
    ${SRC_DIR}/version.h
)

# Gather napkinXC source files
file(GLOB SOURCES
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/utils/*.cpp
    ${SRC_DIR}/blas/*.c
    ${SRC_DIR}/data_readers/*.cpp
    ${SRC_DIR}/liblinear/*.cpp
    ${SRC_DIR}/models/*.cpp)

set(INCLUDES
    ${SRC_DIR}
    ${SRC_DIR}/utils
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

if (PYTHON)
    add_subdirectory(python)
endif ()

if (EXE)
    add_executable(nxc ${SOURCES})
    target_include_directories(nxc PUBLIC ${INCLUDES})
    target_link_libraries(nxc PUBLIC ${LIBRARIES} PRIVATE Threads::Threads)

    add_custom_command(TARGET nxc
        PRE_BUILD
        COMMAND )

    set_target_properties(nxc
        PROPERTIES
        OUTPUT_NAME nxc
        PROJECT_LABEL "napkinXC")
endif ()
