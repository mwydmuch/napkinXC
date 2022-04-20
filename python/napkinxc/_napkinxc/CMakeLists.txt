get_filename_component(CURRENT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
if(${PY_SRC_DIR} STREQUAL ${CURRENT_SRC_DIR})
    set(PY_OUTPUT_DIR ".") # If parent is the base folder
else()
    file(RELATIVE_PATH PY_OUTPUT_DIR ${PY_SRC_DIR} ${CURRENT_SRC_DIR})
endif()

set(PY_SOURCES
    ${SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/_napkinxc.cpp)

pybind11_add_module(pynxc ${PY_SOURCES})
target_include_directories(pynxc PUBLIC ${INCLUDES})
target_link_libraries(pynxc PUBLIC ${LIBRARIES})
set_target_properties(pynxc
        PROPERTIES
        OUTPUT_NAME _napkinxc
        PROJECT_LABEL "napkinXC Python${PYBIND11_PYTHON_VERSION} binding")

if(NOT ${DEPENDENCIES} STREQUAL "")
    add_dependencies(pynxc ${DEPENDENCIES})
endif ()

# Copy all the py files in dir after build
install(TARGETS pynxc DESTINATION ${PY_OUTPUT_DIR} COMPONENT "EXT")


