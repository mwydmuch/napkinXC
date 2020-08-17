# Get pybind11
git_clone(PROJECT_NAME pybind11
    GIT_URL https://github.com/pybind/pybind11.git
    GIT_BRANCH master
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Use included pybind11
add_subdirectory(pybind11)

set(PY_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/napkinxc)
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
file(GLOB PY_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")
foreach( PY_FILE ${PY_FILES} )
    add_custom_command(TARGET pynxc POST_BUILD
        COMMAND cmake -E copy ${PY_FILE} ${PY_OUTPUT_DIR})
endforeach( PY_FILE )
