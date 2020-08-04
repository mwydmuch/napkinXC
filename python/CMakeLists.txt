# Use included pybind11
add_subdirectory(pybind11)

set(PY_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/napkinxc)
set(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PY_SOURCES
    ${PY_SRC_DIR}/_napkinxc.cpp
    ${SOURCES})

pybind11_add_module(pynxc ${PY_SOURCES})
target_include_directories(pynxc PUBLIC ${INCLUDES})
target_link_libraries(pynxc PUBLIC ${LIBRARIES} PRIVATE Threads::Threads)
set_target_properties(pynxc
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${PY_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PY_OUTPUT_DIR}
        OUTPUT_NAME _napkinxc
        PROJECT_LABEL "napkinXC Python${_PYTHON_VERSION} binding")
