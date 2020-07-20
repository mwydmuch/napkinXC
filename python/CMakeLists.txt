#if (BUILD_PYTHON2)
#    find_package(PythonInterp 2 REQUIRED)
#else ()
#    find_package(PythonInterp 3 REQUIRED)
#endif ()
#
#STRING(REGEX REPLACE "([0-9.]+)\\.[0-9]+" "\\1" PYTHON_VERSION ${PYTHON_VERSION_STRING})
#find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)

# Use included pybind11
add_subdirectory(pybind11)

set(PY_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/napkinxc)
set(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PY_SOURCES
    ${PY_SRC_DIR}/pynxc.cpp
    ${SOURCES})

pybind11_add_module(pynxc ${PY_SOURCES})
target_include_directories(pynxc PUBLIC ${INCLUDES})
target_link_libraries(pynxc PUBLIC ${LIBRARIES})
set_target_properties(pynxc
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PY_OUTPUT_DIR}
        OUTPUT_NAME pynxc
        PROJECT_LABEL "napkinXC Python${_PYTHON_VERSION} binding")
