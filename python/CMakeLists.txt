# Set default install path to be dist so CMake by default installs to the same dist directory location as setuptools:
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "default install path" FORCE )
endif()

# Get pybind11
git_clone(PROJECT_NAME pybind11
    GIT_URL https://github.com/pybind/pybind11.git
    GIT_BRANCH master
    #GIT_TAG v2.5.0
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Use included pybind11
add_subdirectory(pybind11)

set(PY_OUTPUT_DIR napkinxc)
#file(RELATIVE_PATH PY_OUTPUT_DIR ${ROOT_DIR} napkinxc_tests)
set(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PY_SOURCES
    ${SOURCES}
    ${PY_SRC_DIR}/_napkinxc.cpp)

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
    OUTPUT_NAME _napkinxc
    PROJECT_LABEL "napkinXC Python${_PYTHON_VERSION} binding")

if(NOT ${DEPENDENCIES} STREQUAL "")
    add_dependencies(pynxc ${DEPENDENCIES})
endif ()

# Copy all the py files in dir after build
file(GLOB PY_FILES LIST_DIRECTORIES false "*.py")

install(FILES ${PY_FILES} DESTINATION ${PY_OUTPUT_DIR} COMPONENT "PY")
install(TARGETS pynxc DESTINATION ${PY_OUTPUT_DIR} COMPONENT "EXT")
