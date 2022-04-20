# Variable to hold the path to src folder to compute relative path of its decendents
set(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Get pybind11
if(NOT EXISTS ${PY_SRC_DIR}/pybind11/CMakeLists.txt)
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        message(STATUS "Pybind11 submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()

    if(NOT EXISTS ${PY_SRC_DIR}/pybind11/CMakeLists.txt)
        message(FATAL_ERROR "The pybind11 submodule was not downloaded!")
    endif()
endif()

find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(pybind11)

add_subdirectory("napkinxc")
