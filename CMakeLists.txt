# napkinXC CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(napkinXC
        VERSION 0.6.1
        DESCRIPTION "Extremely simple and fast extreme multi-class and multi-label classifiers"
        HOMEPAGE_URL https://github.com/mwydmuch/napkinXC
        LANGUAGES C CXX)

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Building options
option(EXE "Build executable" ON)
option(PYTHON "Build Python binding" OFF)
set(PYTHON_VERSION "3" CACHE STRING "Build Python binding with specific Python version")
option(BACKWARD "Build with backward.cpp" OFF)

set(CMAKE_CXX_STANDARD 17)

# Set the release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Configure file
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

configure_file(
        ${SRC_DIR}/version.h.in
        ${SRC_DIR}/version.h)

# Gather napkinXC source files
file(GLOB SOURCES
        ${SRC_DIR}/*.cpp
        ${SRC_DIR}/liblinear/*.cpp
        ${SRC_DIR}/liblinear/blas/*.c
        ${SRC_DIR}/models/*.cpp)

set(INCLUDES
        ${SRC_DIR}
        ${SRC_DIR}/liblinear
        ${SRC_DIR}/liblinear/blas
        ${SRC_DIR}/models)

set(LIBRARIES PRIVATE Threads::Threads)

set(DEPENDENCIES)

if (PYTHON)
    # Set default install path to be dist so CMake by default installs to the same dist directory location as setuptools:
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "default install path" FORCE )
    endif()

    # Get pybind11
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt)
        find_package(Git QUIET)
        if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
            message(STATUS "Pybind11 submodule update")
            file(REMOVE_RECURSE ${PY_SRC_DIR}/pybind11)
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
            if(NOT GIT_SUBMOD_RESULT EQUAL "0")
                message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
            endif()
        endif()

        if(NOT EXISTS ${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt)
            message(FATAL_ERROR "The pybind11 submodule was not downloaded!")
        endif()
    endif()

    if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
        find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development REQUIRED)
    else()
        find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
    endif()

    add_subdirectory(pybind11)
    add_subdirectory(python)
endif ()

if (EXE)
    if (BACKWARD)
        add_subdirectory(${SRC_DIR}/backward)
        add_executable(nxc ${SOURCES} ${BACKWARD_ENABLE})
        add_backward(nxc)
    else()
        add_executable(nxc ${SOURCES})
    endif()
    target_include_directories(nxc PUBLIC ${INCLUDES})
    target_link_libraries(nxc PUBLIC ${LIBRARIES})

    set_target_properties(nxc
            PROPERTIES
            OUTPUT_NAME nxc
            PROJECT_LABEL "napkinXC")

    if(NOT ${DEPENDENCIES} STREQUAL "")
        add_dependencies(nxc ${DEPENDENCIES})
    endif ()
endif ()
