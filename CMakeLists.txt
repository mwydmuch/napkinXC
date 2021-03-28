# napkinXC
# Authors: Marek Wydmuch

cmake_minimum_required(VERSION 3.12)
project(napkinXC
    VERSION 0.5.2
    DESCRIPTION "Extremely simple and fast extreme multi-class and multi-label classifiers"
    HOMEPAGE_URL https://github.com/mwydmuch/napkinXC
    LANGUAGES C CXX)

include(GitUtils.cmake)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Building options
option(EXE "Build executable" ON)
option(PYTHON "Build Python binding" OFF)
option(BACKWARD "Build with backward.cpp" ON)
option(MIPS_EXT "Build extension with Maximum Inner Product Search-based models" OFF)

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
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

configure_file(
    ${SRC_DIR}/version.h.in
    ${SRC_DIR}/version.h
)

# Gather napkinXC source files
file(GLOB SOURCES
    ${SRC_DIR}/*.cpp
    ${SRC_DIR}/blas/*.c
    ${SRC_DIR}/liblinear/*.cpp
    ${SRC_DIR}/models/*.cpp)

set(INCLUDES
    ${SRC_DIR}
    ${SRC_DIR}/blas
    ${SRC_DIR}/liblinear
    ${SRC_DIR}/models)

set(LIBRARIES PRIVATE Threads::Threads)

set(DEPENDENCIES)

# Include backward.cpp
if (BACKWARD)
    list(APPEND SOURCES ${SRC_DIR}/backward/backward.cpp)
    list(APPEND INCLUDES ${SRC_DIR}/backward)
endif ()

# MIPS extension files
if (MIPS_EXT)
    git_clone(PROJECT_NAME nmslib
        GIT_URL https://github.com/nmslib/nmslib.git
        GIT_BRANCH master
        DIRECTORY ${SRC_DIR})

    add_subdirectory(${SRC_DIR}/nmslib/similarity_search)

    link_directories(${SRC_DIR}/nmslib/similarity_search/release)

    file(GLOB MIPS_EXT_SOURCES ${SRC_DIR}/mips_models/*.cpp)
    set(MIPS_EXT_INCLUDES
        ${SRC_DIR}/mips_models
        ${SRC_DIR}/nmslib/similarity_search/include)

    list(APPEND SOURCES ${MIPS_EXT_SOURCES})
    list(APPEND INCLUDES ${MIPS_EXT_INCLUDES})
    list(APPEND LIBRARIES NonMetricSpaceLib)
    list(APPEND DEPENDENCIES NonMetricSpaceLib)
endif ()

if (PYTHON)
    # Set default install path to be dist so CMake by default installs to the same dist directory location as setuptools:
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "default install path" FORCE )
    endif()

    # Get pybind11
    git_clone(PROJECT_NAME pybind11
            GIT_URL https://github.com/pybind/pybind11.git
            #GIT_BRANCH master # To use master version
            GIT_TAG v2.6.2
            DIRECTORY ${ROOT_DIR})

    add_subdirectory("pybind11")
    add_subdirectory(python)
endif ()

if (EXE)
    add_executable(nxc ${SOURCES})
    target_include_directories(nxc PUBLIC ${INCLUDES})
    target_link_libraries(nxc PUBLIC ${LIBRARIES})

    set_target_properties(nxc
        PROPERTIES
        OUTPUT_NAME nxc
#        RUNTIME_OUTPUT_DIRECTORY ${ROOT_DIR}
#        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${ROOT_DIR}
#        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${ROOT_DIR}
#        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${ROOT_DIR}
#        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${ROOT_DIR}
        PROJECT_LABEL "napkinXC")

    if(NOT ${DEPENDENCIES} STREQUAL "")
        add_dependencies(nxc ${DEPENDENCIES})
    endif ()
endif ()
